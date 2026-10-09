#!/usr/bin/env python3
"""Jointly sweep marker and SapBERT gates on fold0 candidate probabilities."""
from copy import deepcopy
from collections import defaultdict
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from transformers import AutoModel,AutoTokenizer
from src.baseline_v1 import LexicalPhenotypeBaseline,build_prediction
from src.data_loader import HPOntology,load_jsonl
from src.hpo_retrieval import LexicalHPORetriever
from src.metrics import compute_all_metrics
raw=ROOT/'data/raw/PatientPheX-V1-A';docs=load_jsonl(str(raw/'PatientPheX-train.jsonl'));by={str(d['pmc_id']):d for d in docs};ids=json.loads((ROOT/'data/processed/e0/splits.json').read_text())['cv_folds'][0]['validation'];gold=[by[x] for x in ids];candidate_path=Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/'data/processed/e14/marker_fold0_joint/candidate_probabilities.jsonl';candidate_rows=[json.loads(x) for x in open(candidate_path)];onto=HPOntology(str(raw/'hp.obo'));lex=LexicalPhenotypeBaseline(onto);ret=LexicalHPORetriever(onto)
def surface(row):
 doc=by[str(row['pmc_id'])]
 for p in doc['full_text']:
  a=row['offset']-p['offset'];b=row['end']-p['offset']
  if 0<=a<b<=len(p['text']):return p['text'][a:b]
 return ''
for row in candidate_rows:row['surface']=surface(row)
name='cambridgeltl/SapBERT-from-PubMedBERT-fulltext';tok=AutoTokenizer.from_pretrained(name);model=AutoModel.from_pretrained(name).to('cuda').eval()
def enc(texts,b=256):
 out=[]
 with torch.inference_mode():
  for i in range(0,len(texts),b):
   x=tok(texts[i:i+b],padding=True,truncation=True,max_length=32,return_tensors='pt').to('cuda');v=model(**x).last_hidden_state[:,0].float();out.append(torch.nn.functional.normalize(v,p=2,dim=1).cpu())
 return torch.cat(out)
phrases=ret.phrases;pv=enc(phrases).to('cuda');surfaces=sorted({ret.normalize(x['surface']) for x in candidate_rows});q=enc(surfaces);scores,indices=(q.to('cuda')@pv.T).max(1);mapping={s:(sorted(ret.phrase_to_ids[phrases[i]])[0],float(z)) for s,i,z in zip(surfaces,indices.cpu().tolist(),scores.cpu().tolist())};by_candidate=defaultdict(list)
for row in candidate_rows:by_candidate[str(row['pmc_id'])].append(row)
runs=[]
for marker_threshold in [.5,.6,.65,.7,.75,.8,.85,.9]:
 for sap_threshold in [.8,.84,.86,.88,.9,.92,.94]:
  predictions=[]
  for doc in gold:
   entities=lex.extract_document(doc,overlap_policy='longest',detect_negation=False);occupied={(x['offset'],x['length']) for x in entities}
   for row in by_candidate[str(doc['pmc_id'])]:
    if row['probability']<marker_threshold:continue
    identifier,score=mapping[ret.normalize(row['surface'])]
    if score<sap_threshold:continue
    key=(row['offset'],row['end']-row['offset'])
    if key in occupied:continue
    entities.append({'identifier':identifier,'type':'Phenotype','offset':key[0],'length':key[1],'text':row['surface'],'note':None});occupied.add(key)
   predictions.append(build_prediction(doc,entities,association_method='same_passage',association_window=0))
  m=compute_all_metrics(gold,predictions);runs.append({'marker_threshold':marker_threshold,'sapbert_threshold':sap_threshold,'metrics':m})
runs.sort(key=lambda x:(x['metrics']['subtask1']['f1_men']+x['metrics']['subtask1']['f1_doc'])/2,reverse=True);report={'scope':'fold0 joint threshold selection only','runs':runs};(ROOT/'data/processed/e14/joint_threshold_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');
for x in runs[:12]:print(x['marker_threshold'],x['sapbert_threshold'],x['metrics']['subtask1']['f1_men'],x['metrics']['subtask1']['f1_doc'],x['metrics']['overall_score'])
