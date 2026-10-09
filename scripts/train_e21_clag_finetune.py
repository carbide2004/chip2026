#!/usr/bin/env python3
"""CLAG-1: end-to-end top-layer fine-tuning with latent Noisy-OR attribution."""
from copy import deepcopy
import argparse,json,random,statistics,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np,torch
from torch import nn
from transformers import AutoModel,AutoTokenizer,get_linear_schedule_with_warmup
from src.data_loader import HPOntology,load_jsonl
from src.metrics import compute_all_metrics
from run_e6_llm_router import relation_prediction
from train_e20_clag0 import build_bags
class Model(nn.Module):
 def __init__(self,name,feature_dim):
  super().__init__();self.encoder=AutoModel.from_pretrained(name);h=self.encoder.config.hidden_size
  layers=getattr(getattr(self.encoder,'encoder',None),'layer',[])
  for layer in list(layers)[:-4]:
   for p in layer.parameters():p.requires_grad=False
  if hasattr(self.encoder,'embeddings'):
   for p in self.encoder.embeddings.parameters():p.requires_grad=False
  self.head=nn.Sequential(nn.LayerNorm(h+feature_dim),nn.Linear(h+feature_dim,256),nn.GELU(),nn.Dropout(.15),nn.Linear(256,1))
 def forward(self,ids,mask,features):
  cls=self.encoder(input_ids=ids,attention_mask=mask).last_hidden_state[:,0];return self.head(torch.cat([cls,features],-1)).squeeze(-1)
def attach_tokens(records,texts,tokenizer,max_length=192):
 enc=tokenizer(texts,padding='max_length',truncation=True,max_length=max_length,return_tensors='pt');bags=[]
 for r in records:
  n=r['occurrences']*r['patients'];start=r['text_start'];item=dict(r);item['input_ids']=enc['input_ids'][start:start+n];item['attention_mask']=enc['attention_mask'][start:start+n];item['features']=torch.tensor(r['features'],dtype=torch.float).reshape(n,-1);bags.append(item)
 return bags
def bag_loss(logits,bag,pos_weight,background_penalty):
 q=logits.reshape(bag['occurrences'],bag['patients']).sigmoid().clamp(1e-5,1-1e-5);relation=(1-torch.prod(1-q,dim=0)).clamp(1e-5,1-1e-5);labels=torch.tensor(bag['labels'],dtype=torch.float,device=logits.device);bce=-(pos_weight*labels*torch.log(relation)+(1-labels)*torch.log(1-relation)).mean();excess=torch.relu(q.sum(1)-1).pow(2).mean();features=bag['features'].to(logits.device).reshape(bag['occurrences'],bag['patients'],-1);background=(features[:,:,10]+features[:,:,11]+features[:,:,13]+features[:,:,14]).clamp_max(1);bg_loss=(q*background).mean();return bce+.05*excess+background_penalty*bg_loss
def infer(model,bags,device):
 output={};model.eval()
 with torch.inference_mode():
  for b in bags:
   logits=model(b['input_ids'].to(device),b['attention_mask'].to(device),b['features'].to(device));q=logits.reshape(b['occurrences'],b['patients']).sigmoid().clamp(1e-5,1-1e-5);probs=(1-torch.prod(1-q,dim=0)).clamp(1e-5,1-1e-5).cpu().tolist()
   for pid,z in zip(b['patient_ids'],probs):output[(b['pmc_id'],pid,b['concept'])]=z
 return output
def main():
 p=argparse.ArgumentParser();p.add_argument('--fold',type=int,default=0);p.add_argument('--seed',type=int,default=2026);p.add_argument('--epochs',type=int,default=3);p.add_argument('--background-penalty',type=float,default=0.0);p.add_argument('--model',default='microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext');p.add_argument('--output-dir',type=Path,default=ROOT/'data/processed/e21/fold0');a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True);random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed);started=time.time();raw=ROOT/'data/raw/PatientPheX-V1-A';docs=load_jsonl(str(raw/'PatientPheX-train.jsonl'));by={str(d['pmc_id']):d for d in docs};split=json.loads((ROOT/'data/processed/e0/splits.json').read_text())['cv_folds'][a.fold];train=[by[x] for x in split['train']];gold=[by[x] for x in split['validation']];entities={str(d['pmc_id']):d['entities'] for d in load_jsonl(str(ROOT/'data/processed/e14/gated_oof'/f'fold{a.fold}.jsonl'))};validation=[]
 for d in gold:x=deepcopy(d);x['entities']=entities[str(d['pmc_id'])];validation.append(x)
 ontology=HPOntology(str(raw/'hp.obo'));train_records,train_texts=build_bags(train,ontology);val_records,val_texts=build_bags(validation,ontology);tokenizer=AutoTokenizer.from_pretrained(a.model);train_bags=attach_tokens(train_records,train_texts,tokenizer);val_bags=attach_tokens(val_records,val_texts,tokenizer);feature_dim=train_bags[0]['features'].shape[-1];device='cuda';model=Model(a.model,feature_dim).to(device);head_params=list(model.head.parameters());head_ids={id(x) for x in head_params};encoder_params=[x for x in model.parameters() if x.requires_grad and id(x) not in head_ids];optimizer=torch.optim.AdamW([{'params':encoder_params,'lr':2e-5},{'params':head_params,'lr':5e-4}],weight_decay=.01);steps=((len(train_bags)+7)//8)*a.epochs;scheduler=get_linear_schedule_with_warmup(optimizer,max(1,steps//10),steps);positives=sum(sum(b['labels']) for b in train_bags);total=sum(len(b['labels']) for b in train_bags);pos_weight=min(10.0,(total-positives)/max(1,positives));history=[]
 for epoch in range(1,a.epochs+1):
  random.shuffle(train_bags);model.train();losses=[]
  for start in range(0,len(train_bags),8):
   chunk=train_bags[start:start+8];ids=torch.cat([b['input_ids'] for b in chunk]).to(device);mask=torch.cat([b['attention_mask'] for b in chunk]).to(device);features=torch.cat([b['features'] for b in chunk]).to(device);optimizer.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.bfloat16):
    logits=model(ids,mask,features);offset=0;parts=[]
    for b in chunk:
     n=b['occurrences']*b['patients'];parts.append(bag_loss(logits[offset:offset+n],b,pos_weight,a.background_penalty));offset+=n
    loss=torch.stack(parts).mean()
   loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step();scheduler.step();losses.append(float(loss.detach()))
  history.append({'epoch':epoch,'loss':statistics.mean(losses)});print(history[-1],flush=True)
 probabilities=infer(model,val_bags,device);keys=list(probabilities);values=[probabilities[k] for k in keys]
 with (a.output_dir/'validation_probabilities.jsonl').open('w') as handle:
  for k,z in probabilities.items():handle.write(json.dumps({'key':list(k),'probability':z})+'\n')
 runs=[]
 for threshold in [.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9]:
  pred=relation_prediction(validation,entities,keys,values,threshold);m=compute_all_metrics(gold,pred);runs.append({'threshold':threshold,'metrics':m});print(threshold,m['overall_score'],m['subtask2']['f1_micro'],m['subtask2']['f1_macro'])
 best=max(runs,key=lambda x:x['metrics']['overall_score']);report={'scope':f'CV fold {a.fold}; locked validation untouched','method':'top-4-layer fine-tuned CLAG Noisy-OR','background_penalty':a.background_penalty,'train_bags':len(train_bags),'validation_bags':len(val_bags),'positive_relations':positives,'relation_count':total,'history':history,'runs':runs,'best':best,'elapsed_seconds':time.time()-started};(a.output_dir/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(best,indent=2))
if __name__=='__main__':main()
