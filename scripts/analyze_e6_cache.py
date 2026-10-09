#!/usr/bin/env python3
"""Compare E6 routing policies using cached Qwen decisions only."""
from collections import defaultdict
from copy import deepcopy
import json,statistics,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from src.baseline_v1 import LexicalPhenotypeBaseline
from src.data_loader import HPOntology,load_jsonl
from src.metrics import compute_all_metrics
from src.relation_features import build_examples
raw=ROOT/'data/raw/PatientPheX-V1-A';docs=load_jsonl(str(raw/'PatientPheX-train.jsonl'));by={str(d['pmc_id']):d for d in docs};splits=json.loads((ROOT/'data/processed/e0/splits.json').read_text());ext=LexicalPhenotypeBaseline(HPOntology(str(raw/'hp.obo')))
def pred(d):x=deepcopy(d);x['entities']=ext.extract_document(d,overlap_policy='longest',detect_negation=False);return x
def cache(fold):
 p=ROOT/'data/processed/e6'/('fold0' if fold==0 else f'fold{fold}_fixed')/'llm_cache.jsonl';return {tuple(r['key']):r for r in map(json.loads,p.read_text().splitlines())}
def make(docs,keys,probs,labels,single,multi,use_llm,uncertain_negative=True,low=.35,high=.85):
 counts={str(d['pmc_id']):len(d['patient']) for d in docs};chosen=defaultdict(list)
 for key,prob in zip(keys,probs):
  threshold=single if counts[key[0]]==1 else multi;decision=prob>=threshold;row=labels.get(tuple(key))
  if use_llm and low<=prob<=high and row:
   if row['label']=='YES':decision=True
   elif row['label']=='NO' or (row['label']=='UNCERTAIN' and uncertain_negative):decision=False
  if decision:chosen[(key[0],key[1])].append(key[2])
 return [{'pmc_id':d['pmc_id'],'pmid':d.get('pmid'),'entities':d['entities'],'association':[{'patient_id':p['patient_id'],'phenotype':chosen[(str(d['pmc_id']),str(p['patient_id']))]} for p in d['patient']]} for d in docs]
policies=[('base_global',.65,.65,False),('base_split',.35,.60,False),('llm_global',.65,.65,True),('llm_split',.35,.60,True)];results={x[0]:[] for x in policies}
for f,fold in enumerate(splits['cv_folds']):
 train=[by[x] for x in fold['train']];val_gold=[by[x] for x in fold['validation']];val=[pred(d) for d in val_gold];xt,yt,_=build_examples(train);xv,_,keys=build_examples(val);m=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000,class_weight='balanced',C=1,random_state=2026)).fit(xt,yt);probs=m.predict_proba(xv)[:,1];labels=cache(f)
 for name,single,multi,use in policies:results[name].append(compute_all_metrics(val_gold,make(val,keys,probs,labels,single,multi,use)))
summary={}
for name,rows in results.items():summary[name]={k:statistics.mean(r['overall_score'] if k=='overall' else r['subtask2'][k] for r in rows) for k in ['overall','f1_micro','f1_macro']}
out={'scope':'four CV folds; cached Qwen decisions; locked validation untouched','policies':{'base_global':{'single':.65,'multi':.65},'base_split':{'single':.35,'multi':.60},'llm_global':{'single':.65,'multi':.65,'route':[.35,.85]},'llm_split':{'single':.35,'multi':.60,'route':[.35,.85]}},'summary':summary,'folds':results};path=ROOT/'data/processed/e6/aggregate_report.json';path.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps(summary,indent=2))
