#!/usr/bin/env python3
"""Calibrate relation thresholds on lexical predicted candidates across CV folds."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import json
from pathlib import Path
import statistics
import sys

REPO_ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO_ROOT))
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from src.baseline_v1 import LexicalPhenotypeBaseline
from src.data_loader import HPOntology,load_jsonl
from src.metrics import evaluate_subtask2
from src.relation_features import build_examples

ROOT=REPO_ROOT/'data/raw/PatientPheX-V1-A';OUT=REPO_ROOT/'data/processed/e5';OUT.mkdir(parents=True,exist_ok=True)
docs=load_jsonl(str(ROOT/'PatientPheX-train.jsonl'));by={str(d['pmc_id']):d for d in docs};splits=json.loads((REPO_ROOT/'data/processed/e0/splits.json').read_text());extractor=LexicalPhenotypeBaseline(HPOntology(str(ROOT/'hp.obo')))

def pred_doc(d):
 x=deepcopy(d);x['entities']=extractor.extract_document(d,overlap_policy='longest',detect_negation=False);return x

def predictions(validation,keys,probabilities,single_threshold,multi_threshold):
 chosen=defaultdict(list);counts={str(d['pmc_id']):len(d['patient']) for d in validation}
 for key,probability in zip(keys,probabilities):
  threshold=single_threshold if counts[key[0]]==1 else multi_threshold
  if probability>=threshold:chosen[(key[0],key[1])].append(key[2])
 return [{'pmc_id':d['pmc_id'],'association':[{'patient_id':p['patient_id'],'phenotype':chosen[(str(d['pmc_id']),str(p['patient_id']))]} for p in d['patient']]} for d in validation]

fold_data=[]
for fold in splits['cv_folds']:
 train=[by[x] for x in fold['train']];validation=[pred_doc(by[x]) for x in fold['validation']]
 xt,yt,_=build_examples(train);xv,_,keys=build_examples(validation);model=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000,class_weight='balanced',C=1.0,random_state=2026)).fit(xt,yt);fold_data.append((validation,keys,model.predict_proba(xv)[:,1]))
thresholds=[x/100 for x in range(30,86,5)];runs=[]
for single in thresholds:
 for multi in thresholds:
  metrics=[]
  for validation,keys,probabilities in fold_data:metrics.append(evaluate_subtask2(validation,predictions(validation,keys,probabilities,single,multi)))
  runs.append({'single_threshold':single,'multi_threshold':multi,'mean_micro':statistics.mean(x['f1_micro'] for x in metrics),'mean_macro':statistics.mean(x['f1_macro'] for x in metrics),'folds':metrics})
runs.sort(key=lambda x:x['mean_micro']+x['mean_macro'],reverse=True)
report={'scope':'four CV folds, lexical candidates, gold-candidate relation training; locked validation untouched','selection':'maximize mean Micro-F1 + Macro-F1','runs':runs};(OUT/'threshold_sweep.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(runs[:10],ensure_ascii=False,indent=2))
