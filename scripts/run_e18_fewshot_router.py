#!/usr/bin/env python3
"""Retrieve fold-training positive/negative demonstrations for Qwen relation routing."""
from collections import defaultdict
from copy import deepcopy
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np,torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM,AutoTokenizer
from src.data_loader import HPOntology,load_jsonl
from src.metrics import compute_all_metrics
from src.relation_features import build_examples,positive_occurrences
from run_e6_llm_router import evidence_prompt,parse_label,relation_prediction,passage_for_offset

def demo(doc,key,label,ontology):
 pid,value=key[1],key[2];patient=next(p for p in doc['patient'] if str(p['patient_id'])==pid);aliases=', '.join(m['text'] for m in patient.get('mention',[])[:4]) or 'none';items=positive_occurrences(doc).get(value,[]);snippet='none';section=''
 if items:
  o=items[0];p=passage_for_offset(doc,o['offset']);local=o['offset']-p['offset'];snippet=' '.join(p['text'][max(0,local-140):min(len(p['text']),local+240)].split());section=p.get('section_type')
 term=ontology.terms.get(value);return f'TARGET {pid}; aliases {aliases}; phenotype {term.name if term else value}; section {section}; evidence {snippet}\nANSWER: {"YES" if label else "NO"}'
def main():
 p=argparse.ArgumentParser();p.add_argument('--fold',type=int,default=0);p.add_argument('--model',default=str(ROOT/'data/models/Qwen3-8B-hf'));p.add_argument('--output-dir',type=Path,default=ROOT/'data/processed/e18/fold0');a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True);raw=ROOT/'data/raw/PatientPheX-V1-A';docs=load_jsonl(str(raw/'PatientPheX-train.jsonl'));by={str(d['pmc_id']):d for d in docs};split=json.loads((ROOT/'data/processed/e0/splits.json').read_text())['cv_folds'][a.fold];train=[by[x] for x in split['train']];gold=[by[x] for x in split['validation']];sup={str(d['pmc_id']):d['entities'] for d in load_jsonl(str(ROOT/'data/processed/e14/gated_oof'/f'fold{a.fold}.jsonl'))};validation=[]
 for d in gold:x=deepcopy(d);x['entities']=sup[str(d['pmc_id'])];validation.append(x)
 xt,yt,kt=build_examples(train);xv,yv,kv=build_examples(validation);scaler=StandardScaler().fit(xt);zt=scaler.transform(xt);zv=scaler.transform(xv);classifier=LogisticRegression(max_iter=1000,class_weight='balanced',C=1,random_state=2026).fit(zt,yt);prob=classifier.predict_proba(zv)[:,1];indices=[i for i,z in enumerate(prob) if .35<=z<=.85];positive=np.where(np.asarray(yt)==1)[0];negative=np.where(np.asarray(yt)==0)[0];ontology=HPOntology(str(raw/'hp.obo'));cache_path=a.output_dir/'llm_cache.jsonl';cache={}
 if cache_path.exists():
  for line in cache_path.read_text().splitlines():row=json.loads(line);cache[tuple(row['key'])]=row
 pending=[i for i in indices if tuple(kv[i]) not in cache];print('routed',len(indices),'pending',len(pending),flush=True);tokenizer=AutoTokenizer.from_pretrained(a.model,padding_side='left');model=AutoModelForCausalLM.from_pretrained(a.model,dtype=torch.bfloat16,device_map='cuda',attn_implementation='sdpa').eval();train_by={str(d['pmc_id']):d for d in train};val_by={str(d['pmc_id']):d for d in validation}
 for start in range(0,len(pending),8):
  batch_ids=pending[start:start+8];prompts=[]
  for i in batch_ids:
   distances=((zt-zv[i])**2).sum(1);pos=positive[np.argmin(distances[positive])];neg=negative[np.argmin(distances[negative])];examples='TRAINING EXAMPLES:\n\n'+demo(train_by[kt[pos][0]],kt[pos],1,ontology)+'\n\n'+demo(train_by[kt[neg][0]],kt[neg],0,ontology);query=evidence_prompt(val_by[kv[i][0]],kv[i][1],kv[i][2],ontology);messages=[{'role':'system','content':'You are a precise biomedical relation classifier. Learn the decision boundary from the labelled examples, then follow the requested output format.'},{'role':'user','content':examples+'\n\nQUERY:\n'+query}];prompts.append(tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False))
  encoded=tokenizer(prompts,return_tensors='pt',padding=True,truncation=True,max_length=3072).to(model.device)
  with torch.inference_mode():generated=model.generate(**encoded,max_new_tokens=4,do_sample=False,pad_token_id=tokenizer.eos_token_id)
  rows=[];length=encoded['input_ids'].shape[1]
  for output,i in zip(generated,batch_ids):
   raw_output=tokenizer.decode(output[length:],skip_special_tokens=True).strip();row={'key':list(kv[i]),'probability':float(prob[i]),'gold_candidate_label':int(yv[i]),'label':parse_label(raw_output),'raw':raw_output};rows.append(row);cache[tuple(kv[i])]=row
  with cache_path.open('a') as handle:
   for row in rows:handle.write(json.dumps(row,ensure_ascii=False)+'\n')
  print('completed',min(start+len(batch_ids),len(pending)),'/',len(pending),flush=True)
 entities={str(d['pmc_id']):d['entities'] for d in validation};pred=relation_prediction(validation,entities,kv,prob,.65,cache,'negative');metrics=compute_all_metrics(gold,pred);queried=[cache[tuple(kv[i])] for i in indices];report={'scope':f'CV fold {a.fold}; retrieved demos from fold training only','metrics':metrics,'routed':len(indices),'label_counts':dict(__import__('collections').Counter(x['label'] for x in queried)),'accuracy':sum((x['label']=='YES')==bool(x['gold_candidate_label']) for x in queried)/len(queried)};(a.output_dir/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
