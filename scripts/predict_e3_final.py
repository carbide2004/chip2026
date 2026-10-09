#!/usr/bin/env python3
"""Predict high-confidence BIO phenotype spans for a supplied unlabeled set."""
from collections import defaultdict
from pathlib import Path
import argparse,sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForTokenClassification,AutoTokenizer
from src.data_loader import load_jsonl,save_jsonl
from train_e3_token_ner import WindowDataset,decode_batch

def main():
 p=argparse.ArgumentParser();p.add_argument('--input-file',type=Path,default=ROOT/'data/raw/PatientPheX-V1-A/PatientPheX-A.jsonl');p.add_argument('--model-dir',type=Path,default=ROOT/'data/processed/final/token_ner/model');p.add_argument('--output-file',type=Path,default=ROOT/'data/processed/final/a_span_predictions.jsonl');p.add_argument('--threshold',type=float,default=.90);a=p.parse_args();docs=load_jsonl(str(a.input_file));tokenizer=AutoTokenizer.from_pretrained(a.model_dir);dataset=WindowDataset(docs,tokenizer,include_metadata=True);loader=DataLoader(dataset,batch_size=8);model=AutoModelForTokenClassification.from_pretrained(a.model_dir).to('cuda').eval();predicted=defaultdict(set)
 with torch.inference_mode():
  for batch in loader:
   logits=model(input_ids=batch['input_ids'].to('cuda'),attention_mask=batch['attention_mask'].to('cuda')).logits.float().cpu();probs=logits.softmax(-1);entity_prob=1-probs[:,:,0];entity_label=probs[:,:,1:].argmax(-1)+1;labels=torch.where(entity_prob>=a.threshold,entity_label,torch.zeros_like(entity_label));decode_batch(labels,batch,predicted)
 out=[]
 for doc in docs:
  entities=[]
  for left,right in sorted(predicted.get(str(doc['pmc_id']),set())):
   for passage in doc['full_text']:
    x=left-passage['offset'];y=right-passage['offset']
    if 0<=x<=y<=len(passage['text']):entities.append({'identifier':'-1','type':'Phenotype','offset':left,'length':right-left,'text':passage['text'][x:y],'note':None});break
  out.append({'pmc_id':doc['pmc_id'],'pmid':doc.get('pmid'),'entities':entities,'association':[{'patient_id':p['patient_id'],'phenotype':[]} for p in doc['patient']]})
 a.output_file.parent.mkdir(parents=True,exist_ok=True);save_jsonl(out,str(a.output_file));print('documents',len(out),'spans',sum(len(d['entities']) for d in out),'output',a.output_file)
if __name__=='__main__':main()
