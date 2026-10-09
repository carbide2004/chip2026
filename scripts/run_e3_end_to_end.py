#!/usr/bin/env python3
"""Normalize E3 span predictions with SapBERT and evaluate fold 0 end-to-end."""

from __future__ import annotations

from collections import defaultdict
import argparse
import json
from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch
from transformers import AutoModel, AutoTokenizer

from src.baseline_v1 import LexicalPhenotypeBaseline, build_prediction
from src.data_loader import HPOntology, load_jsonl, save_jsonl
from src.hpo_retrieval import LexicalHPORetriever
from src.metrics import compute_all_metrics
from src.submission import validate_submission


def encode(texts, tokenizer, model, device, batch_size=256):
    result=[];model.eval()
    with torch.inference_mode():
        for i in range(0,len(texts),batch_size):
            batch=tokenizer(texts[i:i+batch_size],padding=True,truncation=True,max_length=32,return_tensors='pt').to(device)
            vectors=model(**batch).last_hidden_state[:,0].float()
            result.append(torch.nn.functional.normalize(vectors,p=2,dim=1).cpu())
    return torch.cat(result)


def span_score(gold, prediction):
    gold_sets={str(d['pmc_id']):{(e['offset'],e['length']) for e in d.get('entities',[]) if str(e.get('note') or '').upper()!='NO'} for d in gold}
    pred_sets={str(d['pmc_id']):{(e['offset'],e['length']) for e in d.get('entities',[]) if str(e.get('note') or '').upper()!='NO'} for d in prediction}
    tp=fp=fn=0
    for pmc_id in gold_sets:
        g,p=gold_sets[pmc_id],pred_sets.get(pmc_id,set());tp+=len(g&p);fp+=len(p-g);fn+=len(g-p)
    precision=tp/(tp+fp) if tp+fp else 0;recall=tp/(tp+fn) if tp+fn else 0
    return {'precision':precision,'recall':recall,'f1':2*precision*recall/(precision+recall) if precision+recall else 0,'tp':tp,'fp':fp,'fn':fn}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--sapbert',default='cambridgeltl/SapBERT-from-PubMedBERT-fulltext')
    parser.add_argument('--fold',type=int,default=0)
    parser.add_argument('--data-dir',type=Path,default=REPO_ROOT/'data/raw/PatientPheX-V1-A')
    parser.add_argument('--e0-dir',type=Path,default=REPO_ROOT/'data/processed/e0')
    parser.add_argument('--span-dir',type=Path,default=REPO_ROOT/'data/processed/e3/token_ner_fold0')
    parser.add_argument('--span-file',default='span_predictions.jsonl')
    parser.add_argument('--output-dir',type=Path,default=REPO_ROOT/'data/processed/e3/end_to_end_fold0')
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True);started=time.time()

    docs=load_jsonl(str(args.data_dir/'PatientPheX-train.jsonl'));by_id={str(d['pmc_id']):d for d in docs}
    splits=json.loads((args.e0_dir/'splits.json').read_text());val_ids=splits['cv_folds'][args.fold]['validation'];gold=[by_id[x] for x in val_ids]
    span_docs=load_jsonl(str(args.span_dir/args.span_file));span_by_id={str(d['pmc_id']):d for d in span_docs}
    ontology=HPOntology(str(args.data_dir/'hp.obo'));lexical_retriever=LexicalHPORetriever(ontology)
    lexical_extractor=LexicalPhenotypeBaseline(ontology)

    surfaces=sorted({lexical_retriever.normalize(e['text']) for d in span_docs for e in d['entities']})
    device='cuda' if torch.cuda.is_available() else 'cpu';tokenizer=AutoTokenizer.from_pretrained(args.sapbert);model=AutoModel.from_pretrained(args.sapbert).to(device)
    revision=getattr(model.config,'_commit_hash',None)
    phrase_vectors=encode(lexical_retriever.phrases,tokenizer,model,device);query_vectors=encode(surfaces,tokenizer,model,device)
    top_indices=[];phrase_gpu=phrase_vectors.to(device)
    for i in range(0,len(query_vectors),128):top_indices.extend((query_vectors[i:i+128].to(device)@phrase_gpu.T).argmax(dim=1).cpu().tolist())
    surface_to_id={}
    for surface,index in zip(surfaces,top_indices):
        exact=lexical_retriever.phrase_to_ids.get(surface,set())
        if exact:
            canonical={identifier for identifier in exact if lexical_retriever.canonical_names.get(identifier)==surface}
            surface_to_id[surface]=sorted(canonical or exact)[0]
        else:surface_to_id[surface]=sorted(lexical_retriever.phrase_to_ids[lexical_retriever.phrases[index]])[0]

    systems={}
    token_entities={}
    lexical_entities={}
    hybrid_entities={}
    for doc in gold:
        pmc_id=str(doc['pmc_id']);normalized=[]
        for entity in span_by_id[pmc_id]['entities']:
            item=dict(entity);item['identifier']=surface_to_id[lexical_retriever.normalize(item['text'])];normalized.append(item)
        token_entities[pmc_id]=normalized
        lex=lexical_extractor.extract_document(doc,overlap_policy='longest',detect_negation=False)
        lexical_entities[pmc_id]=lex
        exact_spans={(e['offset'],e['length']) for e in lex}
        hybrid_entities[pmc_id]=lex+[e for e in normalized if (e['offset'],e['length']) not in exact_spans]

    for name,entity_map in [('token_dense',token_entities),('lexical',lexical_entities),('hybrid',hybrid_entities)]:
        prediction=[build_prediction(doc,entity_map[str(doc['pmc_id'])],association_method='same_passage',association_window=0) for doc in gold]
        issues=validate_submission(gold,prediction)
        systems[name]={'span':span_score(gold,prediction),'metrics':compute_all_metrics(gold,prediction),'validation_issues':[x.to_dict() for x in issues]}
        save_jsonl(prediction,str(args.output_dir/f'{name}.jsonl'))
    report={'scope':f'CV fold {args.fold} only; locked validation untouched','sapbert':{'name':args.sapbert,'revision':revision},'elapsed_seconds':time.time()-started,'systems':systems}
    (args.output_dir/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
