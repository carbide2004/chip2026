#!/usr/bin/env python3
"""Multi-fold E15 relation hyperparameter search; never reads locked validation."""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
from collections import defaultdict
import itertools,json,statistics,sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from src.data_loader import load_jsonl
from src.metrics import compute_all_metrics
from src.relation_features import build_examples
from run_e6_llm_router import load_cache,relation_prediction
CS=[.003,.01,.02,.03,.05,.07,.1]
WEIGHT_MULTIPLIERS=[.5,.6,.7,.8,.9]
THRESHOLDS=[.68,.7,.72,.74,.75,.76,.78,.8,.82]
LOWS=[.15,.2,.25,.3,.35]
HIGHS=[.75,.85,.95]
MODES=['rescue','override']
def decisions(keys,probs,cache,threshold,low,high,mode):
 rows={}
 for key,p in zip(map(tuple,keys),probs):
  base=p>=threshold;label=cache.get(key,{}).get('label');active=low<=p<=high and label in {'YES','NO'}
  if mode=='override' and active:value=label=='YES'
  elif mode=='veto' and active and label=='NO':value=False
  elif mode=='rescue' and active and label=='YES':value=True
  else:value=base
  rows[key]={'label':'YES' if value else 'NO'}
 return rows
def main():
 raw=ROOT/'data/raw/PatientPheX-V1-A';docs=load_jsonl(str(raw/'PatientPheX-train.jsonl'));by={str(d['pmc_id']):d for d in docs};splits=json.loads((ROOT/'data/processed/e0/splits.json').read_text())['cv_folds'];folds=[]
 for fold,split in enumerate(splits):
  train=[by[x] for x in split['train']];gold=[by[x] for x in split['validation']];entities={str(d['pmc_id']):d['entities'] for d in load_jsonl(str(ROOT/f'data/processed/e14/gated_oof/fold{fold}.jsonl'))};validation=[]
  for d in gold:x=deepcopy(d);x['entities']=entities[str(d['pmc_id'])];validation.append(x)
  xt,yt,_=build_examples(train);xv,_,keys=build_examples(validation);cache=load_cache(ROOT/f'data/processed/e15/fold{fold}/llm_cache.jsonl');folds.append((gold,validation,entities,xt,np.asarray(yt),xv,keys,cache))
 results=[]
 for c,wm in itertools.product(CS,WEIGHT_MULTIPLIERS):
  probabilities=[]
  for gold,validation,entities,xt,yt,xv,keys,cache in folds:
   ratio=(len(yt)-yt.sum())/max(1,yt.sum());model=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,class_weight={0:1.,1:ratio*wm},C=c,random_state=2026));probabilities.append(model.fit(xt,yt).predict_proba(xv)[:,1])
  for threshold,low,high,mode in itertools.product(THRESHOLDS,LOWS,HIGHS,MODES):
   if low>high:continue
   metrics=[]
   for data,probs in zip(folds,probabilities):
    gold,validation,entities,xt,yt,xv,keys,cache=data;rows=decisions(keys,probs,cache,threshold,low,high,mode);pred=relation_prediction(validation,entities,keys,probs,threshold,rows,'negative');metrics.append(compute_all_metrics(gold,pred))
   scores=[m['overall_score'] for m in metrics];results.append({'C':c,'weight_multiplier':wm,'threshold':threshold,'low':low,'high':high,'mode':mode,'mean_overall':statistics.mean(scores),'min_overall':min(scores),'fold_overall':scores,'mean_micro':statistics.mean(m['subtask2']['f1_micro'] for m in metrics),'mean_macro':statistics.mean(m['subtask2']['f1_macro'] for m in metrics)})
 results.sort(key=lambda x:(x['mean_overall'],x['min_overall']),reverse=True);baseline=[json.loads((ROOT/f'data/processed/e15/fold{i}/report.json').read_text())['metrics']['llm_uncertain_as_negative']['overall_score'] for i in range(4)];report={'scope':'four-fold development OOF only; locked validation untouched','grid':{'C':CS,'weight_multiplier':WEIGHT_MULTIPLIERS,'threshold':THRESHOLDS,'low':LOWS,'high':HIGHS,'mode':MODES},'baseline':{'fold_overall':baseline,'mean_overall':statistics.mean(baseline)},'top':results[:100]};out=ROOT/'data/processed/e15/tuning_report_fine.json';out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print('baseline',report['baseline']);
 for row in results[:20]:print({k:(round(v,6) if isinstance(v,float) else [round(x,6) for x in v] if isinstance(v,list) else v) for k,v in row.items()})
if __name__=='__main__':main()
