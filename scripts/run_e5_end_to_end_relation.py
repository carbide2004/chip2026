#!/usr/bin/env python3
"""Evaluate evidence-bag relation models on predicted lexical entities."""

from __future__ import annotations

from copy import deepcopy
import argparse
import json
from pathlib import Path
import statistics
import sys
import time

REPO_ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO_ROOT))

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.baseline_v1 import LexicalPhenotypeBaseline,build_prediction
from src.data_loader import HPOntology,load_jsonl
from src.metrics import compute_all_metrics
from src.relation_features import build_examples


def with_entities(doc,entities):
 result=deepcopy(doc);result['entities']=entities;return result


def predict_associations(docs,keys,probabilities,threshold):
 selected={}
 for (pmc_id,patient_id,value),probability in zip(keys,probabilities):
  if probability>=threshold:selected.setdefault((pmc_id,patient_id),[]).append(value)
 return [{'pmc_id':d['pmc_id'],'pmid':d.get('pmid'),'entities':d['entities'],'association':[{'patient_id':p['patient_id'],'phenotype':selected.get((str(d['pmc_id']),str(p['patient_id'])),[])} for p in d['patient']]} for d in docs]


def main():
 p=argparse.ArgumentParser();p.add_argument('--threshold',type=float,default=.65);p.add_argument('--data-dir',type=Path,default=REPO_ROOT/'data/raw/PatientPheX-V1-A');p.add_argument('--e0-dir',type=Path,default=REPO_ROOT/'data/processed/e0');p.add_argument('--output-dir',type=Path,default=REPO_ROOT/'data/processed/e5');args=p.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
 docs=load_jsonl(str(args.data_dir/'PatientPheX-train.jsonl'));by_id={str(d['pmc_id']):d for d in docs};splits=json.loads((args.e0_dir/'splits.json').read_text());extractor=LexicalPhenotypeBaseline(HPOntology(str(args.data_dir/'hp.obo')));started=time.time();results=[]
 def predicted_doc(doc):return with_entities(doc,extractor.extract_document(doc,overlap_policy='longest',detect_negation=False))
 for fold in splits['cv_folds']:
  train_gold=[by_id[x] for x in fold['train']];validation_gold=[by_id[x] for x in fold['validation']];train_pred=[predicted_doc(d) for d in train_gold];validation_pred=[predicted_doc(d) for d in validation_gold]
  systems={}
  systems['same_passage']=[build_prediction(d,d['entities'],association_method='same_passage',association_window=0) for d in validation_pred]
  for mode,training_docs in [('gold_candidate_training',train_gold),('predicted_candidate_training',train_pred)]:
   x_train,y_train,_=build_examples(training_docs);x_val,_,keys=build_examples(validation_pred)
   model=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000,class_weight='balanced',C=1.0,random_state=2026));model.fit(x_train,y_train);prob=model.predict_proba(x_val)[:,1]
   systems[mode]=predict_associations(validation_pred,keys,prob,args.threshold)
  fold_result={'fold':fold['fold'],'systems':{name:compute_all_metrics(validation_gold,pred) for name,pred in systems.items()}}
  results.append(fold_result)
 summary={}
 for name in results[0]['systems']:
  summary[name]={key:statistics.mean((r['systems'][name]['overall_score'] if key=='overall_score' else r['systems'][name]['subtask2'][key]) for r in results) for key in ['overall_score','f1_micro','f1_macro']}
 report={'scope':'lexical predicted entities over four CV folds; locked validation untouched','threshold':args.threshold,'threshold_origin':'selected from E4 gold-entity CV; E5 results are not an independent threshold test','elapsed_seconds':time.time()-started,'summary':summary,'folds':results};(args.output_dir/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
