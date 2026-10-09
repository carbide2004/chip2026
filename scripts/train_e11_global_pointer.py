#!/usr/bin/env python3
"""Minimal GlobalPointer-style span NER experiment on one CV fold."""
from __future__ import annotations
from collections import defaultdict
import argparse,json,math,random,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from transformers import AutoModel,AutoTokenizer,get_linear_schedule_with_warmup
from src.data_loader import load_jsonl,save_jsonl
from train_e3_token_ner import positive_spans
class SpanDataset(Dataset):
 def __init__(self,docs,tokenizer,max_length=512,stride=128,max_width=24,max_pairs=128,metadata=False):
  self.items=[];self.gold=defaultdict(set);self.max_width=max_width;self.dropped_width=0
  for doc in docs:
   pmc=str(doc['pmc_id'])
   for passage in doc.get('full_text',[]):
    text=passage.get('text','');spans=positive_spans(doc,passage)
    for a,b in spans:self.gold[pmc].add((passage['offset']+a,passage['offset']+b))
    if not text:continue
    enc=tokenizer(text,truncation=True,max_length=max_length,stride=stride,return_overflowing_tokens=True,return_offsets_mapping=True,padding='max_length')
    for ids,mask,offsets in zip(enc['input_ids'],enc['attention_mask'],enc['offset_mapping']):
     content=[x for x in offsets if tuple(x)!=(0,0)]
     if not content:continue
     lo,hi=min(x[0] for x in content),max(x[1] for x in content);pairs=[]
     for a,b in spans:
      if a<lo or b>hi:continue
      idx=[i for i,(x,y) in enumerate(offsets) if (x,y)!=(0,0) and x<b and y>a]
      if not idx:continue
      width=idx[-1]-idx[0]
      if width>=max_width:self.dropped_width+=1;continue
      pairs.append((idx[0],width))
     pair_tensor=torch.full((max_pairs,2),-1,dtype=torch.long)
     for i,pair in enumerate(pairs[:max_pairs]):pair_tensor[i]=torch.tensor(pair)
     content_mask=[tuple(x)!=(0,0) for x in offsets];tokens=tokenizer.convert_ids_to_tokens(ids);start_mask=[valid and not token.startswith('##') for valid,token in zip(content_mask,tokens)];end_mask=[valid and (i+1>=len(tokens) or not content_mask[i+1] or not tokens[i+1].startswith('##')) for i,valid in enumerate(content_mask)]
     item={'input_ids':torch.tensor(ids),'attention_mask':torch.tensor(mask),'content_mask':torch.tensor(content_mask),'start_mask':torch.tensor(start_mask),'end_mask':torch.tensor(end_mask),'positive_pairs':pair_tensor}
     if metadata:item.update({'pmc_id':pmc,'passage_offset':passage['offset'],'offset_mapping':torch.tensor(offsets)})
     self.items.append(item)
 def __len__(self):return len(self.items)
 def __getitem__(self,i):return self.items[i]
class SpanModel(nn.Module):
 def __init__(self,name,max_width=24,dim=96):
  super().__init__();self.encoder=AutoModel.from_pretrained(name);h=self.encoder.config.hidden_size;self.start=nn.Linear(h,dim);self.end=nn.Linear(h,dim);self.width_bias=nn.Parameter(torch.zeros(max_width));self.max_width=max_width;self.scale=math.sqrt(dim)
 def forward(self,input_ids,attention_mask):
  h=self.encoder(input_ids=input_ids,attention_mask=attention_mask).last_hidden_state;s=self.start(h);e=self.end(h);b,l,d=s.shape;logits=s.new_full((b,l,self.max_width),-30)
  for w in range(self.max_width):logits[:,:l-w,w]=(s[:,:l-w]*e[:,w:]).sum(-1)/self.scale+self.width_bias[w]
  return logits
def loss_fn(logits,batch):
 b,l,w=logits.shape;target=torch.zeros_like(logits);pairs=batch['positive_pairs'].to(logits.device)
 for row in range(b):
  valid=pairs[row,:,0]>=0;starts=pairs[row,valid,0];widths=pairs[row,valid,1];target[row,starts,widths]=1
 starts=batch['start_mask'].to(logits.device);ends=batch['end_mask'].to(logits.device);mask=torch.zeros_like(target,dtype=torch.bool)
 for width in range(w):mask[:,:l-width,width]=starts[:,:l-width]&ends[:,width:]
 values=target[mask];scores=logits[mask];pos_weight=30.0;return nn.functional.binary_cross_entropy_with_logits(scores,values,pos_weight=torch.tensor(pos_weight,device=logits.device))
def predict(model,loader,threshold,device):
 output=defaultdict(set);model.eval()
 with torch.inference_mode():
  for batch in loader:
   scores=model(batch['input_ids'].to(device),batch['attention_mask'].to(device)).sigmoid().cpu();hits=scores>=threshold;starts=batch['start_mask'];ends=batch['end_mask']
   for width in range(scores.shape[2]):hits[:,:scores.shape[1]-width,width]&=starts[:,:scores.shape[1]-width]&ends[:,width:]
   for r in range(scores.shape[0]):
    offsets=batch['offset_mapping'][r].tolist();base=int(batch['passage_offset'][r]);pmc=batch['pmc_id'][r]
    for start,width in hits[r].nonzero().tolist():
     end=start+width
     if end>=len(offsets) or tuple(offsets[start])==(0,0) or tuple(offsets[end])==(0,0):continue
     output[pmc].add((base+offsets[start][0],base+offsets[end][1]))
 return output
def metric(gold,pred):
 tp=fp=fn=0
 for k in set(gold)|set(pred):g=gold.get(k,set());p=pred.get(k,set());tp+=len(g&p);fp+=len(p-g);fn+=len(g-p)
 precision=tp/(tp+fp) if tp+fp else 0;recall=tp/(tp+fn) if tp+fn else 0;f1=2*precision*recall/(precision+recall) if precision+recall else 0;return {'precision':precision,'recall':recall,'f1':f1,'tp':tp,'fp':fp,'fn':fn}
def main():
 p=argparse.ArgumentParser();p.add_argument('--fold',type=int,default=0);p.add_argument('--epochs',type=int,default=5);p.add_argument('--batch-size',type=int,default=4);p.add_argument('--model',default='microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext');p.add_argument('--output-dir',type=Path,default=ROOT/'data/processed/e11/fold0');a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True);seed=2026;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);device='cuda';raw=ROOT/'data/raw/PatientPheX-V1-A';docs=load_jsonl(str(raw/'PatientPheX-train.jsonl'));by={str(d['pmc_id']):d for d in docs};splits=json.loads((ROOT/'data/processed/e0/splits.json').read_text())['cv_folds'][a.fold];train=[by[x] for x in splits['train']];val=[by[x] for x in splits['validation']];tok=AutoTokenizer.from_pretrained(a.model,use_fast=True);tr=SpanDataset(train,tok);va=SpanDataset(val,tok,metadata=True);loader=DataLoader(tr,batch_size=a.batch_size,shuffle=True);vloader=DataLoader(va,batch_size=a.batch_size);model=SpanModel(a.model).to(device);opt=torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay=.01);steps=len(loader)*a.epochs;sched=get_linear_schedule_with_warmup(opt,max(1,steps//10),steps);history=[];started=time.time()
 for epoch in range(1,a.epochs+1):
  model.train();losses=[]
  for batch in loader:
   opt.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(batch['input_ids'].to(device),batch['attention_mask'].to(device));loss=loss_fn(logits,batch)
   loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();sched.step();losses.append(float(loss.detach()))
  row={'epoch':epoch,'loss':sum(losses)/len(losses)};history.append(row);print(row,flush=True)
 sweep=[]
 for threshold in [.1,.2,.3,.4,.5,.6,.7,.8,.9,.95,.975,.99,.995,.999]:
  pred=predict(model,vloader,threshold,device);m=metric(va.gold,pred);sweep.append({'threshold':threshold,**m});print(sweep[-1],flush=True)
 best=max(sweep,key=lambda x:x['f1']);pred=predict(model,vloader,best['threshold'],device);rows=[]
 for doc in val:
  entities=[]
  for left,right in sorted(pred.get(str(doc['pmc_id']),set())):
   for passage in doc['full_text']:
    x=left-passage['offset'];y=right-passage['offset']
    if 0<=x<y<=len(passage['text']):entities.append({'identifier':'-1','type':'Phenotype','offset':left,'length':right-left,'text':passage['text'][x:y],'note':None});break
  rows.append({'pmc_id':doc['pmc_id'],'pmid':doc.get('pmid'),'entities':entities,'association':[]})
 save_jsonl(rows,str(a.output_dir/'span_predictions.jsonl'));report={'scope':f'CV fold {a.fold}; locked validation untouched','history':history,'sweep':sweep,'best':best,'train_windows':len(tr),'validation_windows':len(va),'dropped_width':{'train':tr.dropped_width,'validation':va.dropped_width},'elapsed_seconds':time.time()-started};(a.output_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
