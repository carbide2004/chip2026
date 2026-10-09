#!/usr/bin/env python3
"""E23: align relation training with cross-fitted predicted entity bags."""
from copy import deepcopy
from pathlib import Path
import json,statistics,sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from src.data_loader import load_jsonl
from src.metrics import compute_all_metrics
from src.relation_features import build_examples
from run_e6_llm_router import load_cache,relation_prediction

def replace_entities(docs,entity_map):
 out=[]
 for doc in docs:
  row=deepcopy(doc);row['entities']=deepcopy(entity_map[str(doc['pmc_id'])]);out.append(row)
 return out

def main():
 raw=ROOT/'data/raw/PatientPheX-V1-A';docs=load_jsonl(str(raw/'PatientPheX-train.jsonl'));by={str(x['pmc_id']):x for x in docs};splits=json.loads((ROOT/'data/processed/e0/splits.json').read_text())['cv_folds'];oof={}
 for fold in range(4):
  for row in load_jsonl(str(ROOT/f'data/processed/e14/gated_oof/fold{fold}.jsonl')):oof[str(row['pmc_id'])]=row['entities']
 modes=['gold','predicted','mixed'];fold_results={mode:[] for mode in modes}
 for fold,split in enumerate(splits):
  train_gold=[by[x] for x in split['train']];train_pred=replace_entities(train_gold,oof);gold=[by[x] for x in split['validation']];val_entities={str(x['pmc_id']):x['entities'] for x in load_jsonl(str(ROOT/f'data/processed/e14/gated_oof/fold{fold}.jsonl'))};validation=replace_entities(gold,val_entities);xg,yg,_=build_examples(train_gold);xp,yp,_=build_examples(train_pred);xv,_,keys=build_examples(validation);cache=load_cache(ROOT/f'data/processed/e15/fold{fold}/llm_cache.jsonl')
  datasets={'gold':(xg,yg),'predicted':(xp,yp),'mixed':(xg+xp,yg+yp)}
  for mode,(xt,yt) in datasets.items():
   model=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,class_weight='balanced',C=1,random_state=2026));probs=model.fit(xt,yt).predict_proba(xv)[:,1];pred=relation_prediction(validation,val_entities,keys,probs,.65,cache,'negative');metrics=compute_all_metrics(gold,pred);fold_results[mode].append({'fold':fold,'train_examples':len(yt),'train_positive':sum(yt),'metrics':metrics})
 report={'scope':'four-fold development OOF; relation train entities cross-fitted per document; locked validation untouched','modes':{},'caveat':'base entity model for some relation-train documents may include outer relation-validation entity labels, but no relation labels'}
 for mode,rows in fold_results.items():
  report['modes'][mode]={'folds':rows,'mean':{'overall':statistics.mean(x['metrics']['overall_score'] for x in rows),'micro':statistics.mean(x['metrics']['subtask2']['f1_micro'] for x in rows),'macro':statistics.mean(x['metrics']['subtask2']['f1_macro'] for x in rows)}};print(mode,report['modes'][mode]['mean'],[round(x['metrics']['overall_score'],6) for x in rows])
 out=ROOT/'data/processed/e23';out.mkdir(parents=True,exist_ok=True);(out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
if __name__=='__main__':main()
