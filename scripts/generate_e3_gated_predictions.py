#!/usr/bin/env python3
"""Generate OOF gated entity predictions for all four development folds."""
from __future__ import annotations
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from transformers import AutoModel,AutoTokenizer
from src.baseline_v1 import LexicalPhenotypeBaseline
from src.data_loader import HPOntology,load_jsonl,save_jsonl
from src.hpo_retrieval import LexicalHPORetriever
from src.submission import validate_submission,raise_for_errors
raw=ROOT/'data/raw/PatientPheX-V1-A';out=ROOT/'data/processed/e3/gated_oof';out.mkdir(parents=True,exist_ok=True)
docs=load_jsonl(str(raw/'PatientPheX-train.jsonl'));by={str(d['pmc_id']):d for d in docs};splits=json.loads((ROOT/'data/processed/e0/splits.json').read_text());onto=HPOntology(str(raw/'hp.obo'));lex=LexicalPhenotypeBaseline(onto);ret=LexicalHPORetriever(onto);device='cuda' if torch.cuda.is_available() else 'cpu';model_name='cambridgeltl/SapBERT-from-PubMedBERT-fulltext';tok=AutoTokenizer.from_pretrained(model_name);model=AutoModel.from_pretrained(model_name).to(device).eval()
def enc(texts,b=256):
 result=[]
 with torch.inference_mode():
  for i in range(0,len(texts),b):
   batch=tok(texts[i:i+b],padding=True,truncation=True,max_length=32,return_tensors='pt').to(device);v=model(**batch).last_hidden_state[:,0].float();result.append(torch.nn.functional.normalize(v,p=2,dim=1).cpu())
 return torch.cat(result)
phrases=ret.phrases;phrase_vectors=enc(phrases).to(device);manifest={'sapbert':model_name,'sapbert_revision':getattr(model.config,'_commit_hash',None),'ner_probability_threshold':.90,'sapbert_similarity_threshold':.90,'folds':[]}
for fold in range(4):
 ids=splits['cv_folds'][fold]['validation'];source=[by[x] for x in ids];span_path=ROOT/'data/processed/e3'/f'token_ner_fold{fold}'/'span_predictions_calibrated.jsonl';spans={str(d['pmc_id']):d['entities'] for d in load_jsonl(str(span_path))};surfaces=sorted({ret.normalize(e['text']) for es in spans.values() for e in es});queries=enc(surfaces);scores,indices=(queries.to(device)@phrase_vectors.T).max(dim=1);mapping={surface:(sorted(ret.phrase_to_ids[phrases[index]])[0],float(score)) for surface,index,score in zip(surfaces,indices.cpu().tolist(),scores.cpu().tolist())};predictions=[];added=0
 for_doc=[]
 for doc in source:
  lexical=lex.extract_document(doc,overlap_policy='longest',detect_negation=False);occupied={(e['offset'],e['length']) for e in lexical};entities=list(lexical)
  for entity in spans[str(doc['pmc_id'])]:
   if (entity['offset'],entity['length']) in occupied:continue
   identifier,score=mapping[ret.normalize(entity['text'])]
   if score>=.90:
    item=dict(entity);item['identifier']=identifier;entities.append(item);added+=1
  entities.sort(key=lambda e:(e['offset'],e['length'],e['identifier']))
  predictions.append({'pmc_id':doc['pmc_id'],'pmid':doc.get('pmid'),'entities':entities,'association':[{'patient_id':p['patient_id'],'phenotype':[]} for p in doc['patient']]})
 issues=validate_submission(source,predictions);raise_for_errors(issues);path=out/f'fold{fold}.jsonl';save_jsonl(predictions,str(path));manifest['folds'].append({'fold':fold,'documents':len(predictions),'learned_additions':added,'validation_warnings':[x.to_dict() for x in issues]});print('fold',fold,'docs',len(predictions),'learned additions',added)
(out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n');print(out/'manifest.json')
