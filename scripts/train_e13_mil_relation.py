#!/usr/bin/env python3
"""Occurrence-level attention MIL with full/evidence fusion for patient-HPO relation."""
from collections import defaultdict
from copy import deepcopy
import argparse,json,random,statistics,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np,torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from transformers import AutoModel,AutoTokenizer,get_linear_schedule_with_warmup
from src.data_loader import HPOntology,load_jsonl
from src.metrics import compute_all_metrics
from src.relation_features import build_examples,split_value
from run_e6_llm_router import passage_for_offset,relation_prediction

def instance_texts(doc,patient_id,value,ontology,max_instances=5):
 patient=next(p for p in doc['patient'] if str(p['patient_id'])==patient_id);target=patient.get('mention',[]);target_offsets=[int(m['offset']) for m in target];aliases=', '.join(f'{m["text"]}@{m["offset"]}' for m in target) or 'none';others=[]
 for p in doc['patient']:
  if str(p['patient_id'])!=patient_id:others.append(f'{p["patient_id"]}='+','.join(m['text'] for m in p.get('mention',[])[:3]))
 term=ontology.terms.get(value);prefix=f'TARGET {patient_id}; aliases {aliases}. OTHERS {"; ".join(others) or "none"}. PHENOTYPE {value} {term.name if term else ""}. '
 rows=[]
 for entity in doc.get('entities',[]):
  if value not in split_value(entity.get('identifier'),entity.get('text','')):continue
  passage=passage_for_offset(doc,entity['offset'])
  if not passage:continue
  local=entity['offset']-passage['offset'];left=max(0,local-260);right=min(len(passage['text']),local+entity['length']+260);snippet=' '.join(passage['text'][left:right].split());distance=min((entity['offset']-x for x in target_offsets),key=abs) if target_offsets else 999999;rows.append((abs(distance),prefix+f'SECTION {passage.get("section_type")}; TARGET_DISTANCE {distance:+d}. EVIDENCE {snippet}'))
 rows.sort(key=lambda x:x[0]);return [x[1] for x in rows[:max_instances]] or [prefix+'NO OCCURRENCE']

class BagDataset(Dataset):
 def __init__(self,docs,keys,labels,ontology,tokenizer,max_instances=5,max_length=192):
  by={str(d['pmc_id']):d for d in docs};self.items=[]
  for key,label in zip(keys,labels):
   texts=instance_texts(by[key[0]],key[1],key[2],ontology,max_instances);encoded=tokenizer(texts,padding='max_length',truncation=True,max_length=max_length);n=len(texts);ids=torch.zeros((max_instances,max_length),dtype=torch.long);mask=torch.zeros_like(ids);ids[:n]=torch.tensor(encoded['input_ids']);mask[:n]=torch.tensor(encoded['attention_mask']);self.items.append({'input_ids':ids,'attention_mask':mask,'instance_mask':torch.tensor([1]*n+[0]*(max_instances-n),dtype=torch.bool),'label':torch.tensor(label,dtype=torch.long)})
 def __len__(self):return len(self.items)
 def __getitem__(self,i):return self.items[i]

class MILModel(nn.Module):
 def __init__(self,name):
  super().__init__();self.encoder=AutoModel.from_pretrained(name);h=self.encoder.config.hidden_size;self.evidence_score=nn.Linear(h,1);self.instance_score=nn.Linear(h,1);self.full_proj=nn.Sequential(nn.Linear(h,h),nn.Tanh());self.evidence_proj=nn.Sequential(nn.Linear(h,h),nn.Tanh());self.classifier=nn.Linear(h*2+2,2)
 def forward(self,input_ids,attention_mask,instance_mask):
  b,n,l=input_ids.shape;flat_ids=input_ids.view(b*n,l);flat_mask=attention_mask.view(b*n,l);hidden=self.encoder(input_ids=flat_ids,attention_mask=flat_mask).last_hidden_state[:,0].view(b,n,-1);valid=instance_mask
  full=(hidden*valid.unsqueeze(-1)).sum(1)/valid.sum(1,keepdim=True).clamp_min(1);attention=self.evidence_score(hidden).squeeze(-1).masked_fill(~valid,-1e4).softmax(-1);evidence=(hidden*attention.unsqueeze(-1)).sum(1);instance=self.instance_score(hidden).squeeze(-1).masked_fill(~valid,-1e4).max(1).values;count=valid.sum(1).float().log1p();features=torch.cat([self.full_proj(full),self.evidence_proj(evidence),instance.unsqueeze(-1),count.unsqueeze(-1)],-1);return self.classifier(features),attention

def main():
 p=argparse.ArgumentParser();p.add_argument('--fold',type=int,default=0);p.add_argument('--epochs',type=int,default=3);p.add_argument('--seed',type=int,default=2026);p.add_argument('--model',default='microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext');p.add_argument('--output-dir',type=Path,default=ROOT/'data/processed/e13/fold0');a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True);random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed);raw=ROOT/'data/raw/PatientPheX-V1-A';docs=load_jsonl(str(raw/'PatientPheX-train.jsonl'));by={str(d['pmc_id']):d for d in docs};split=json.loads((ROOT/'data/processed/e0/splits.json').read_text())['cv_folds'][a.fold];train=[by[x] for x in split['train']];gold=[by[x] for x in split['validation']];sup={str(d['pmc_id']):d['entities'] for d in load_jsonl(str(ROOT/'data/processed/e3/gated_oof'/f'fold{a.fold}.jsonl'))};validation=[]
 for doc in gold:item=deepcopy(doc);item['entities']=sup[str(doc['pmc_id'])];validation.append(item)
 _,yt,kt=build_examples(train);_,yv,kv=build_examples(validation);ontology=HPOntology(str(raw/'hp.obo'));tokenizer=AutoTokenizer.from_pretrained(a.model);train_set=BagDataset(train,kt,yt,ontology,tokenizer);val_set=BagDataset(validation,kv,yv,ontology,tokenizer);loader=DataLoader(train_set,batch_size=4,shuffle=True);vloader=DataLoader(val_set,batch_size=8);device='cuda';model=MILModel(a.model).to(device);counts=np.bincount(np.asarray(yt),minlength=2);weights=torch.tensor([len(yt)/(2*max(1,x)) for x in counts],device=device,dtype=torch.float);optimizer=torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay=.01);steps=len(loader)*a.epochs;scheduler=get_linear_schedule_with_warmup(optimizer,max(1,steps//10),steps);history=[];started=time.time()
 for epoch in range(1,a.epochs+1):
  model.train();losses=[]
  for batch in loader:
   optimizer.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.bfloat16):logits,_=model(batch['input_ids'].to(device),batch['attention_mask'].to(device),batch['instance_mask'].to(device));loss=nn.functional.cross_entropy(logits,batch['label'].to(device),weight=weights)
   loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step();scheduler.step();losses.append(float(loss.detach()))
  history.append({'epoch':epoch,'loss':statistics.mean(losses)});print(history[-1],flush=True)
 model.eval();prob=[];attention_rows=[]
 with torch.inference_mode():
  for batch in vloader:
   logits,attention=model(batch['input_ids'].to(device),batch['attention_mask'].to(device),batch['instance_mask'].to(device));prob.extend(logits.softmax(-1)[:,1].float().cpu().tolist());attention_rows.extend(attention.float().cpu().tolist())
 with (a.output_dir/'validation_probabilities.jsonl').open('w',encoding='utf-8') as handle:
  for key,label,prediction,attention in zip(kv,yv,prob,attention_rows):handle.write(json.dumps({'key':list(key),'label':int(label),'probability':float(prediction),'evidence_attention':attention},ensure_ascii=False)+'\n')
 entities={str(d['pmc_id']):d['entities'] for d in validation};runs=[]
 for threshold in [.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85]:
  prediction=relation_prediction(validation,entities,kv,prob,threshold);metrics=compute_all_metrics(gold,prediction);runs.append({'threshold':threshold,'metrics':metrics});print(threshold,metrics['overall_score'],metrics['subtask2']['f1_micro'],metrics['subtask2']['f1_macro'])
 best=max(runs,key=lambda x:x['metrics']['overall_score']);report={'scope':f'CV fold {a.fold}; locked validation untouched','method':'occurrence attention MIL + full/evidence/max fusion + learned NA','train_pairs':len(train_set),'validation_pairs':len(val_set),'class_counts':counts.tolist(),'history':history,'runs':runs,'best':best,'elapsed_seconds':time.time()-started};(a.output_dir/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(best,indent=2))
if __name__=='__main__':main()
