#!/usr/bin/env python3
"""CLAG-0: latent occurrence attribution trained by patient-HPO Noisy-OR likelihood."""
from __future__ import annotations
from collections import defaultdict
from copy import deepcopy
import argparse,json,math,random,statistics,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np,torch
from torch import nn
from transformers import AutoModel,AutoTokenizer
from src.data_loader import HPOntology,load_jsonl
from src.metrics import compute_all_metrics
from src.relation_features import association_set,passage_index,positive_occurrences
from run_e6_llm_router import passage_for_offset,relation_prediction
SECTIONS=['CASE','RESULTS','ABSTRACT','INTRO','DISCUSS','FIG','METHODS','CONCL']
def occurrence_pair_text(doc,patient,concept,occurrence,ontology):
 aliases=', '.join(m['text'] for m in patient.get('mention',[])[:6]) or 'none';others=[]
 for p in doc['patient']:
  if str(p['patient_id'])!=str(patient['patient_id']):others.extend(m['text'] for m in p.get('mention',[])[:2])
 passage=doc['full_text'][occurrence['passage_index']] if occurrence['passage_index'] is not None else passage_for_offset(doc,occurrence['offset']);local=occurrence['offset']-passage['offset'];left=max(0,local-260);right=min(len(passage['text']),local+260);raw=passage['text'];end=min(len(raw),local+int(occurrence.get('length',len(occurrence.get('text','')))));snippet=' '.join((raw[left:local]+' [unused1] '+raw[local:end]+' [unused2] '+raw[end:right]).split());term=ontology.terms.get(concept);return f'TARGET {patient["patient_id"]}; ALIASES {aliases}; OTHER PATIENT ALIASES {", ".join(others) or "none"}; PHENOTYPE {term.name if term else concept}; SECTION {passage.get("section_type")}; CONTEXT {snippet}'
def structural(doc,patient,occurrence):
 pid=str(patient['patient_id']);anchors=[int(m['offset']) for m in patient.get('mention',[])];all_anchors={str(p['patient_id']):[int(m['offset']) for m in p.get('mention',[])] for p in doc['patient']};distance=min((abs(occurrence['offset']-x) for x in anchors),default=10**9);pidx=occurrence['passage_index'];same=any(passage_index(doc,x)==pidx for x in anchors);by={k:min((abs(occurrence['offset']-x) for x in values),default=10**9) for k,values in all_anchors.items()};nearest=by.get(pid,10**9)==min(by.values(),default=10**9);before=occurrence['offset']<min(anchors,default=-1);section=occurrence['section'];return [math.log1p(distance)/22,float(same),float(nearest),float(before),math.log1p(len(anchors))/4,math.log1p(len(doc['patient']))/4,float(len(doc['patient'])==1),*[float(name in section) for name in SECTIONS]]
def build_bags(docs,ontology):
 records=[];texts=[]
 for doc in docs:
  gold={str(a['patient_id']):association_set(a) for a in doc.get('association',[])};occurrences=positive_occurrences(doc);patients=doc['patient']
  for concept,items in occurrences.items():
   if len(items)>20:items=sorted(items,key=lambda o:(0 if any(x in o['section'] for x in ['CASE','RESULTS','ABSTRACT']) else 1,o['offset']))[:20]
   start=len(texts);features=[]
   for occurrence in items:
    row=[]
    for patient in patients:texts.append(occurrence_pair_text(doc,patient,concept,occurrence,ontology));row.append(structural(doc,patient,occurrence))
    features.append(row)
   labels=[int(concept in gold.get(str(p['patient_id']),set())) for p in patients];records.append({'pmc_id':str(doc['pmc_id']),'concept':concept,'patient_ids':[str(p['patient_id']) for p in patients],'occurrences':len(items),'patients':len(patients),'text_start':start,'features':features,'labels':labels})
 return records,texts
def encode(texts,model_name,batch=64):
 tokenizer=AutoTokenizer.from_pretrained(model_name);encoder=AutoModel.from_pretrained(model_name).to('cuda').eval();rows=[]
 with torch.inference_mode():
  for i in range(0,len(texts),batch):
   x=tokenizer(texts[i:i+batch],padding=True,truncation=True,max_length=192,return_tensors='pt').to('cuda');rows.append(encoder(**x).last_hidden_state[:,0].float().cpu())
 del encoder;torch.cuda.empty_cache();return torch.cat(rows) if rows else torch.empty((0,768))
class Scorer(nn.Module):
 def __init__(self,dim):super().__init__();self.net=nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,256),nn.GELU(),nn.Dropout(.15),nn.Linear(256,64),nn.GELU(),nn.Linear(64,1))
 def forward(self,x):return self.net(x).squeeze(-1)
def attach(records,embeddings):
 output=[]
 for r in records:
  n=r['occurrences']*r['patients'];e=embeddings[r['text_start']:r['text_start']+n].reshape(r['occurrences'],r['patients'],-1);f=torch.tensor(r['features'],dtype=torch.float);item=dict(r);item['x']=torch.cat([e,f],-1);output.append(item)
 return output
def relation_probs(model,bags,device):
 output={}
 model.eval()
 with torch.inference_mode():
  for b in bags:
   q=model(b['x'].to(device)).sigmoid().clamp(1e-5,1-1e-5);prob=(1-torch.prod(1-q,dim=0)).clamp(1e-5,1-1e-5)
   for pid,z in zip(b['patient_ids'],prob.cpu().tolist()):output[(b['pmc_id'],pid,b['concept'])]=z
 return output
def main():
 p=argparse.ArgumentParser();p.add_argument('--fold',type=int,default=0);p.add_argument('--seed',type=int,default=2026);p.add_argument('--epochs',type=int,default=40);p.add_argument('--model',default='microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext');p.add_argument('--output-dir',type=Path,default=ROOT/'data/processed/e20/fold0');a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True);random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed);started=time.time();raw=ROOT/'data/raw/PatientPheX-V1-A';docs=load_jsonl(str(raw/'PatientPheX-train.jsonl'));by={str(d['pmc_id']):d for d in docs};split=json.loads((ROOT/'data/processed/e0/splits.json').read_text())['cv_folds'][a.fold];train=[by[x] for x in split['train']];gold=[by[x] for x in split['validation']];entities={str(d['pmc_id']):d['entities'] for d in load_jsonl(str(ROOT/'data/processed/e14/gated_oof'/f'fold{a.fold}.jsonl'))};validation=[]
 for d in gold:x=deepcopy(d);x['entities']=entities[str(d['pmc_id'])];validation.append(x)
 ontology=HPOntology(str(raw/'hp.obo'));train_records,train_texts=build_bags(train,ontology);val_records,val_texts=build_bags(validation,ontology);all_embeddings=encode(train_texts+val_texts,a.model);train_bags=attach(train_records,all_embeddings[:len(train_texts)]);val_bags=attach(val_records,all_embeddings[len(train_texts):]);device='cuda';model=Scorer(train_bags[0]['x'].shape[-1]).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-3);positives=sum(sum(b['labels']) for b in train_bags);total=sum(len(b['labels']) for b in train_bags);pos_weight=min(10.0,(total-positives)/max(1,positives));history=[]
 for epoch in range(1,a.epochs+1):
  random.shuffle(train_bags);model.train();losses=[]
  for b in train_bags:
   optimizer.zero_grad(set_to_none=True);q=model(b['x'].to(device)).sigmoid().clamp(1e-5,1-1e-5);relation=(1-torch.prod(1-q,dim=0)).clamp(1e-5,1-1e-5);labels=torch.tensor(b['labels'],dtype=torch.float,device=device);bce=-(pos_weight*labels*torch.log(relation)+(1-labels)*torch.log(1-relation)).mean();excess=torch.relu(q.sum(1)-1).pow(2).mean();loss=bce+.05*excess;loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step();losses.append(float(loss.detach()))
  if epoch==1 or epoch%5==0:history.append({'epoch':epoch,'loss':statistics.mean(losses)});print(history[-1],flush=True)
 probabilities=relation_probs(model,val_bags,device);keys=list(probabilities);values=[probabilities[k] for k in keys]
 with (a.output_dir/'validation_probabilities.jsonl').open('w',encoding='utf-8') as handle:
  for key,value in probabilities.items():handle.write(json.dumps({'key':list(key),'probability':float(value)},ensure_ascii=False)+'\n')
 runs=[]
 for threshold in [.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9]:
  pred=relation_prediction(validation,entities,keys,values,threshold);m=compute_all_metrics(gold,pred);runs.append({'threshold':threshold,'metrics':m});print(threshold,m['overall_score'],m['subtask2']['f1_micro'],m['subtask2']['f1_macro'])
 best=max(runs,key=lambda x:x['metrics']['overall_score']);report={'scope':f'CV fold {a.fold}; locked validation untouched','method':'frozen BiomedBERT occurrence-patient embeddings + soft Noisy-OR','train_bags':len(train_bags),'validation_bags':len(val_bags),'train_texts':len(train_texts),'validation_texts':len(val_texts),'positive_relations':positives,'relation_count':total,'pos_weight':pos_weight,'history':history,'runs':runs,'best':best,'elapsed_seconds':time.time()-started};(a.output_dir/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(best,indent=2))
if __name__=='__main__':main()
