#!/usr/bin/env python3
"""Normalize E14 marker spans with SapBERT and evaluate fold0 thresholds."""
from copy import deepcopy
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from transformers import AutoModel,AutoTokenizer
from src.baseline_v1 import build_prediction
from src.data_loader import HPOntology,load_jsonl
from src.hpo_retrieval import LexicalHPORetriever
from src.metrics import compute_all_metrics
raw=ROOT/'data/raw/PatientPheX-V1-A';docs=load_jsonl(str(raw/'PatientPheX-train.jsonl'));by={str(d['pmc_id']):d for d in docs};fold=0;ids=json.loads((ROOT/'data/processed/e0/splits.json').read_text())['cv_folds'][fold]['validation'];gold=[by[x] for x in ids];rows={str(d['pmc_id']):d for d in load_jsonl(str(ROOT/'data/processed/e14/marker_fold0/span_predictions.jsonl'))};onto=HPOntology(str(raw/'hp.obo'));ret=LexicalHPORetriever(onto);name='cambridgeltl/SapBERT-from-PubMedBERT-fulltext';tok=AutoTokenizer.from_pretrained(name);model=AutoModel.from_pretrained(name).to('cuda').eval()
def enc(texts,b=256):
 out=[]
 with torch.inference_mode():
  for i in range(0,len(texts),b):
   x=tok(texts[i:i+b],padding=True,truncation=True,max_length=32,return_tensors='pt').to('cuda');v=model(**x).last_hidden_state[:,0].float();out.append(torch.nn.functional.normalize(v,p=2,dim=1).cpu())
 return torch.cat(out)
phrases=ret.phrases;pv=enc(phrases).to('cuda');surfaces=sorted({ret.normalize(e['text']) for d in rows.values() for e in d['entities'] if e['identifier']=='-1'});q=enc(surfaces);scores,indices=(q.to('cuda')@pv.T).max(dim=1);mapping={s:(sorted(ret.phrase_to_ids[phrases[i]])[0],float(z)) for s,i,z in zip(surfaces,indices.cpu().tolist(),scores.cpu().tolist())};runs=[]
for threshold in [.7,.75,.8,.82,.84,.86,.88,.9,.92,.94,.96]:
 predictions=[]
 for doc in gold:
  entities=[]
  for e in rows[str(doc['pmc_id'])]['entities']:
   item=deepcopy(e)
   if item['identifier']=='-1':
    identifier,score=mapping[ret.normalize(item['text'])]
    if score<threshold:continue
    item['identifier']=identifier
   entities.append(item)
  predictions.append(build_prediction(doc,entities,association_method='same_passage',association_window=0))
 m=compute_all_metrics(gold,predictions);runs.append({'sapbert_threshold':threshold,'metrics':m});print(threshold,m['overall_score'],m['subtask1']['f1_men'],m['subtask1']['f1_doc'])
report={'scope':'fold0 only; choose SapBERT gate here then freeze for folds1-3','marker_threshold':.8,'runs':runs,'best':max(runs,key=lambda x:(x['metrics']['subtask1']['f1_men']+x['metrics']['subtask1']['f1_doc'])/2)};path=ROOT/'data/processed/e14/fold0_normalization_report.json';path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report['best'],indent=2))
