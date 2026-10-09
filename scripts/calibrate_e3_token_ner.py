#!/usr/bin/env python3
"""Calibrate E3 token entity probability on its CV validation fold."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

REPO_ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO_ROOT));sys.path.insert(0,str(REPO_ROOT/'scripts'))

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForTokenClassification,AutoTokenizer

from src.data_loader import load_jsonl,save_jsonl
from train_e3_token_ner import WindowDataset,decode_batch


def metrics(gold,pred):
 tp=fp=fn=0
 for pmc_id in set(gold)|set(pred):
  g,p=gold.get(pmc_id,set()),pred.get(pmc_id,set());tp+=len(g&p);fp+=len(p-g);fn+=len(g-p)
 precision=tp/(tp+fp) if tp+fp else 0;recall=tp/(tp+fn) if tp+fn else 0
 return {'precision':precision,'recall':recall,'f1':2*precision*recall/(precision+recall) if precision+recall else 0,'tp':tp,'fp':fp,'fn':fn}


def main():
 p=argparse.ArgumentParser();p.add_argument('--fold',type=int,default=0);p.add_argument('--locked-validation',action='store_true');p.add_argument('--batch-size',type=int,default=8);p.add_argument('--fixed-threshold',type=float)
 p.add_argument('--data-dir',type=Path,default=REPO_ROOT/'data/raw/PatientPheX-V1-A');p.add_argument('--e0-dir',type=Path,default=REPO_ROOT/'data/processed/e0')
 p.add_argument('--model-dir',type=Path,default=REPO_ROOT/'data/processed/e3/token_ner_fold0/model');p.add_argument('--output-dir',type=Path,default=REPO_ROOT/'data/processed/e3/token_ner_fold0')
 args=p.parse_args();docs=load_jsonl(str(args.data_dir/'PatientPheX-train.jsonl'));by_id={str(d['pmc_id']):d for d in docs};split=json.loads((args.e0_dir/'splits.json').read_text());validation_ids=split['locked_validation'] if args.locked_validation else split['cv_folds'][args.fold]['validation'];val=[by_id[x] for x in validation_ids]
 tokenizer=AutoTokenizer.from_pretrained(args.model_dir);dataset=WindowDataset(val,tokenizer,include_metadata=True);loader=DataLoader(dataset,batch_size=args.batch_size)
 device='cuda' if torch.cuda.is_available() else 'cpu';model=AutoModelForTokenClassification.from_pretrained(args.model_dir).to(device).eval();cached=[]
 with torch.inference_mode():
  for batch in loader:
   logits=model(input_ids=batch['input_ids'].to(device),attention_mask=batch['attention_mask'].to(device)).logits.float().cpu()
   cached.append((logits,batch))
 rows=[];best=None;best_pred=None
 thresholds=[args.fixed_threshold] if args.fixed_threshold is not None else [x/100 for x in range(30,96,5)]
 for threshold in thresholds:
  predicted=defaultdict(set)
  for logits,batch in cached:
   probs=logits.softmax(-1);entity_prob=1-probs[:,:,0];entity_label=probs[:,:,1:].argmax(-1)+1
   labels=torch.where(entity_prob>=threshold,entity_label,torch.zeros_like(entity_label));decode_batch(labels,batch,predicted)
  row={'threshold':threshold,**metrics(dataset.gold_spans,predicted)};rows.append(row)
  if best is None or row['f1']>best['f1']:best=row;best_pred=predicted
 report={'mode':'fixed' if args.fixed_threshold is not None else 'selected_on_this_fold','best':best,'sweep':rows}
 print(json.dumps(report,indent=2));(args.output_dir/'calibration_report.json').write_text(json.dumps(report,indent=2)+'\n')
 prediction_docs=[]
 for doc in val:
  entities=[]
  for left,right in sorted(best_pred.get(str(doc['pmc_id']),set())):
   for passage in doc['full_text']:
    local=left-passage['offset'];end=right-passage['offset']
    if 0<=local<=end<=len(passage['text']):
     entities.append({'identifier':'-1','type':'Phenotype','offset':left,'length':right-left,'text':passage['text'][local:end],'note':None});break
  prediction_docs.append({'pmc_id':doc['pmc_id'],'pmid':doc.get('pmid'),'entities':entities,'association':[{'patient_id':x['patient_id'],'phenotype':[]} for x in doc['patient']]})
 save_jsonl(prediction_docs,str(args.output_dir/'span_predictions_calibrated.jsonl'))

if __name__=='__main__':main()
