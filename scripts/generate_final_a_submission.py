#!/usr/bin/env python3
"""Fit frozen relation model on all 80 docs and create the A submission."""
from collections import defaultdict
from copy import deepcopy
import argparse,json,re,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM,AutoTokenizer
from src.data_loader import HPOntology,load_jsonl,save_jsonl
from src.relation_features import build_examples
from src.submission import validate_submission,raise_for_errors
from run_e6_llm_router import evidence_prompt,parse_label,load_cache,append_cache
parser=argparse.ArgumentParser();parser.add_argument('--train-file',type=Path,default=ROOT/'data/raw/PatientPheX-V1-A/PatientPheX-train.jsonl');parser.add_argument('--input-file',type=Path,default=ROOT/'data/raw/PatientPheX-V1-A/PatientPheX-A.jsonl');parser.add_argument('--ontology-file',type=Path,default=ROOT/'data/raw/PatientPheX-V1-A/hp.obo');parser.add_argument('--qwen-model',type=Path,default=ROOT/'data/models/Qwen3-8B-hf');parser.add_argument('--entities-file',type=Path,default=ROOT/'data/processed/final/a_gated_entities.jsonl');parser.add_argument('--cache-file',type=Path,default=ROOT/'data/processed/final/a_llm_cache.jsonl');parser.add_argument('--output-file',type=Path,default=ROOT/'submissions/patientphex_a_frozen_e7.jsonl');parser.add_argument('--summary-file',type=Path,default=ROOT/'data/processed/final/a_submission_summary.json');parser.add_argument('--C',type=float,default=1.0);parser.add_argument('--positive-weight-multiplier',type=float,default=None);parser.add_argument('--low',type=float,default=.35);parser.add_argument('--high',type=float,default=.85);parser.add_argument('--threshold',type=float,default=.65);parser.add_argument('--qwen-policy',choices=['override','rescue'],default='override');args=parser.parse_args()
train=load_jsonl(str(args.train_file));a_source=load_jsonl(str(args.input_file));entity_rows={str(d['pmc_id']):d['entities'] for d in load_jsonl(str(args.entities_file))};a_docs=[]
for d in a_source:x=deepcopy(d);x['entities']=entity_rows[str(d['pmc_id'])];a_docs.append(x)
xt,yt,_=build_examples(train);xv,_,keys=build_examples(a_docs);class_weight='balanced'
if args.positive_weight_multiplier is not None:
 ratio=(len(yt)-sum(yt))/max(1,sum(yt));class_weight={0:1.0,1:ratio*args.positive_weight_multiplier}
classifier=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,class_weight=class_weight,C=args.C,random_state=2026)).fit(xt,yt);probabilities=classifier.predict_proba(xv)[:,1];low,high,threshold=args.low,args.high,args.threshold;indices=[i for i,p in enumerate(probabilities) if low<=p<=high];out_dir=ROOT/'data/processed/final';cache_path=args.cache_file;cache=load_cache(cache_path);pending=[i for i in indices if tuple(keys[i]) not in cache];ontology=HPOntology(str(args.ontology_file));model_path=args.qwen_model;print('pairs',len(keys),'routed',len(indices),'cached',len(indices)-len(pending),'pending',len(pending),flush=True)
if pending:
 tokenizer=AutoTokenizer.from_pretrained(model_path,padding_side='left');model=AutoModelForCausalLM.from_pretrained(model_path,dtype=torch.bfloat16,device_map='cuda',attn_implementation='sdpa').eval();by={str(d['pmc_id']):d for d in a_docs}
 for start in range(0,len(pending),8):
  batch_indices=pending[start:start+8];prompts=[]
  for i in batch_indices:
   pmc,patient,value=keys[i];messages=[{'role':'system','content':'You are a precise biomedical relation classifier. Follow the requested output format.'},{'role':'user','content':evidence_prompt(by[pmc],patient,value,ontology)}];prompts.append(tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False))
  encoded=tokenizer(prompts,return_tensors='pt',padding=True,truncation=True,max_length=2048).to(model.device)
  with torch.inference_mode():generated=model.generate(**encoded,max_new_tokens=4,do_sample=False,pad_token_id=tokenizer.eos_token_id)
  rows=[];input_length=encoded['input_ids'].shape[1]
  for output,i in zip(generated,batch_indices):
   raw_output=tokenizer.decode(output[input_length:],skip_special_tokens=True).strip();row={'key':list(keys[i]),'probability':float(probabilities[i]),'label':parse_label(raw_output),'raw':raw_output};rows.append(row);cache[tuple(keys[i])]=row
  append_cache(cache_path,rows);print('completed',min(start+len(batch_indices),len(pending)),'/',len(pending),flush=True)
selected=defaultdict(list)
for key,probability in zip(keys,probabilities):
 decision=probability>=threshold;row=cache.get(tuple(key))
 if row and low<=probability<=high:
  if row['label']=='YES':decision=True
  elif args.qwen_policy=='override':decision=False
 if decision:selected[(key[0],key[1])].append(key[2])
submission=[]
for doc in a_docs:
 pmc=str(doc['pmc_id']);submission.append({'pmc_id':doc['pmc_id'],'pmid':doc.get('pmid'),'entities':doc['entities'],'association':[{'patient_id':p['patient_id'],'phenotype':selected[(pmc,str(p['patient_id']))]} for p in doc['patient']]})
issues=validate_submission(a_source,submission);raise_for_errors(issues);path=args.output_file;save_jsonl(submission,str(path));summary={'relation_config':{'C':args.C,'positive_weight_multiplier':args.positive_weight_multiplier,'low':low,'high':high,'threshold':threshold,'qwen_policy':args.qwen_policy},'documents':len(submission),'entities':sum(len(d['entities']) for d in submission),'patients':sum(len(d['association']) for d in submission),'associations':sum(len(a['phenotype']) for d in submission for a in d['association']),'routed':len(indices),'llm_labels':dict(__import__('collections').Counter(cache[tuple(keys[i])]['label'] for i in indices)),'validation_warnings':[x.to_dict() for x in issues],'output':str(path)};args.summary_file.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');print(json.dumps(summary,ensure_ascii=False,indent=2))
