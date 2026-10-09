#!/usr/bin/env python3
"""Train patient-concept evidence-bag classifiers using gold entity occurrences."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time

REPO_ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO_ROOT))

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.baseline_v1 import build_prediction
from src.data_loader import load_jsonl
from src.metrics import evaluate_subtask2
from src.relation_features import build_examples,feature_names


def prediction_docs(docs,keys,probabilities,threshold):
 by_patient=defaultdict(list)
 for (pmc_id,patient_id,value),probability in zip(keys,probabilities):
  if probability>=threshold:by_patient[(pmc_id,patient_id)].append(value)
 output=[]
 for doc in docs:
  output.append({'pmc_id':doc['pmc_id'],'association':[{'patient_id':p['patient_id'],'phenotype':by_patient[(str(doc['pmc_id']),str(p['patient_id']))]} for p in doc['patient']]})
 return output


def main():
 p=argparse.ArgumentParser();p.add_argument('--data-dir',type=Path,default=REPO_ROOT/'data/raw/PatientPheX-V1-A');p.add_argument('--e0-dir',type=Path,default=REPO_ROOT/'data/processed/e0');p.add_argument('--output-dir',type=Path,default=REPO_ROOT/'data/processed/e4');args=p.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
 docs=load_jsonl(str(args.data_dir/'PatientPheX-train.jsonl'));by_id={str(d['pmc_id']):d for d in docs};splits=json.loads((args.e0_dir/'splits.json').read_text());started=time.time();results=[]
 for fold in splits['cv_folds']:
  train=[by_id[x] for x in fold['train']];validation=[by_id[x] for x in fold['validation']]
  x_train,y_train,_=build_examples(train);x_val,y_val,keys=build_examples(validation)
  models={
   'logistic':make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000,class_weight='balanced',C=1.0,random_state=2026)),
   'hist_gradient_boosting':HistGradientBoostingClassifier(max_iter=200,learning_rate=.05,max_depth=4,l2_regularization=1.0,class_weight='balanced',random_state=2026),
  }
  for name,model in models.items():
   model.fit(x_train,y_train);probabilities=model.predict_proba(x_val)[:,1];thresholds=[]
   for threshold in [x/100 for x in range(10,91,5)]:
    metrics=evaluate_subtask2(validation,prediction_docs(validation,keys,probabilities,threshold))
    thresholds.append({'threshold':threshold,**metrics})
   row={'fold':fold['fold'],'model':name,'train_examples':len(x_train),'validation_examples':len(x_val),'positive_rate_train':sum(y_train)/len(y_train),'thresholds':thresholds}
   if name=='logistic':row['coefficients']=dict(zip(feature_names(),model[-1].coef_[0].tolist()))
   results.append(row)
  baseline=[build_prediction(doc,doc['entities'],association_method='same_passage',association_window=0) for doc in validation]
  results.append({'fold':fold['fold'],'model':'same_passage','metrics':evaluate_subtask2(validation,baseline)})
 report={'scope':'gold entities; four CV folds; locked validation untouched','association_composite_semantics':'provisional semicolon splitting','elapsed_seconds':time.time()-started,'results':results}
 (args.output_dir/'relation_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
 for fold in range(4):
  print('fold',fold)
  for row in [x for x in results if x['fold']==fold]:
   if row['model']=='same_passage':print(' ',row['model'],round(row['metrics']['f1_micro'],4),round(row['metrics']['f1_macro'],4))
   else:
    best=max(row['thresholds'],key=lambda x:x['f1_micro']+x['f1_macro']);print(' ',row['model'],'best-on-fold',best['threshold'],round(best['f1_micro'],4),round(best['f1_macro'],4))
 print('report',args.output_dir/'relation_report.json')

if __name__=='__main__':main()
