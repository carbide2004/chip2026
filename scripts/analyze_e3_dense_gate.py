#!/usr/bin/env python3
"""Tune a SapBERT similarity gate for learned NER additions on folds 0-1."""
from collections import defaultdict
import json,statistics,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from transformers import AutoModel,AutoTokenizer
from src.baseline_v1 import LexicalPhenotypeBaseline,build_prediction
from src.data_loader import HPOntology,load_jsonl
from src.hpo_retrieval import LexicalHPORetriever
from src.metrics import compute_all_metrics
raw=ROOT/'data/raw/PatientPheX-V1-A';docs=load_jsonl(str(raw/'PatientPheX-train.jsonl'));by={str(d['pmc_id']):d for d in docs};splits=json.loads((ROOT/'data/processed/e0/splits.json').read_text());onto=HPOntology(str(raw/'hp.obo'));lex=LexicalPhenotypeBaseline(onto);ret=LexicalHPORetriever(onto);device='cuda';tok=AutoTokenizer.from_pretrained('cambridgeltl/SapBERT-from-PubMedBERT-fulltext');model=AutoModel.from_pretrained('cambridgeltl/SapBERT-from-PubMedBERT-fulltext').to(device).eval()
def enc(texts,b=256):
 out=[]
 with torch.inference_mode():
  for i in range(0,len(texts),b):
   x=tok(texts[i:i+b],padding=True,truncation=True,max_length=32,return_tensors='pt').to(device);v=model(**x).last_hidden_state[:,0].float();out.append(torch.nn.functional.normalize(v,p=2,dim=1).cpu())
 return torch.cat(out)
phrases=ret.phrases;pv=enc(phrases).to(device)
fold_data=[]
for fold in [0,1,2,3]:
 ids=splits['cv_folds'][fold]['validation'];gold=[by[x] for x in ids];span_dir=ROOT/'data/processed/e3'/f'token_ner_fold{fold}';spans={str(d['pmc_id']):d['entities'] for d in load_jsonl(str(span_dir/'span_predictions_calibrated.jsonl'))};surfaces=sorted({ret.normalize(e['text']) for es in spans.values() for e in es});qv=enc(surfaces);scores,indices=(qv.to(device)@pv.T).max(dim=1);mapping={s:(sorted(ret.phrase_to_ids[phrases[i]])[0],float(score)) for s,i,score in zip(surfaces,indices.cpu().tolist(),scores.cpu().tolist())};fold_data.append((gold,spans,mapping))
thresholds=[x/100 for x in range(70,100,2)];runs=[]
for threshold in thresholds:
 folds=[]
 for gold,spans,mapping in fold_data:
  predictions=[]
  for doc in gold:
   lexical=lex.extract_document(doc,overlap_policy='longest',detect_negation=False);occupied={(e['offset'],e['length']) for e in lexical};entities=list(lexical)
   for e in spans[str(doc['pmc_id'])]:
    if (e['offset'],e['length']) in occupied:continue
    identifier,score=mapping[ret.normalize(e['text'])]
    if score>=threshold:item=dict(e);item['identifier']=identifier;entities.append(item)
   predictions.append(build_prediction(doc,entities,association_method='same_passage',association_window=0))
  folds.append(compute_all_metrics(gold,predictions))
 runs.append({'threshold':threshold,'mean_task1':statistics.mean(r['subtask1']['f1_men']+r['subtask1']['f1_doc'] for r in folds)/2,'mean_overall':statistics.mean(r['overall_score'] for r in folds),'folds':folds})
runs.sort(key=lambda x:x['mean_task1'],reverse=True);out={'scope':'four CV folds; locked validation untouched','selection':'maximize mean of Mention F1 and Document F1','runs':runs};path=ROOT/'data/processed/e3/dense_gate_report.json';path.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps(runs[:8],ensure_ascii=False,indent=2))
