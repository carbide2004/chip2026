#!/usr/bin/env python3
"""Train frozen Marker on labelled data and generate E15 entities for an input set."""
from collections import defaultdict
import argparse,json,random,statistics,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import numpy as np,torch
from torch.utils.data import DataLoader
from transformers import AutoModel,AutoModelForSequenceClassification,AutoTokenizer,get_linear_schedule_with_warmup
from src.baseline_v1 import LexicalPhenotypeBaseline
from src.data_loader import HPOntology,load_jsonl,save_jsonl
from src.hpo_retrieval import LexicalHPORetriever
from src.submission import validate_submission,raise_for_errors
from train_e14_marker_ner import MarkerDataset,bio_spans,candidates

def main():
 p=argparse.ArgumentParser();p.add_argument('--train-file',type=Path,default=ROOT/'data/raw/PatientPheX-V1-A/PatientPheX-train.jsonl');p.add_argument('--input-file',type=Path,default=ROOT/'data/raw/PatientPheX-V1-A/PatientPheX-A.jsonl');p.add_argument('--ontology-file',type=Path,default=ROOT/'data/raw/PatientPheX-V1-A/hp.obo');p.add_argument('--bio-model',type=Path,default=ROOT/'data/processed/final/token_ner/model');p.add_argument('--output-file',type=Path,default=ROOT/'data/processed/final/a_e14_entities.jsonl');p.add_argument('--report-file',type=Path,default=ROOT/'data/processed/final/e14_report.json');p.add_argument('--marker-model-dir',type=Path,default=None);p.add_argument('--reuse-marker-model',action='store_true');p.add_argument('--marker-threshold',type=float,default=.60);p.add_argument('--sapbert-threshold',type=float,default=.90);p.add_argument('--seed',type=int,default=2026);a=p.parse_args();a.output_file.parent.mkdir(parents=True,exist_ok=True);a.report_file.parent.mkdir(parents=True,exist_ok=True);random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed);started=time.time();train=load_jsonl(str(a.train_file));test=load_jsonl(str(a.input_file));ontology=HPOntology(str(a.ontology_file));lexical=LexicalPhenotypeBaseline(ontology);train_bio=bio_spans(train,a.bio_model);test_bio=bio_spans(test,a.bio_model);_,train_examples=candidates(train,train_bio,lexical);_,test_examples=candidates(test,test_bio,lexical);model_name='microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext';tokenizer=AutoTokenizer.from_pretrained(model_name);train_set=MarkerDataset(train_examples,tokenizer);test_set=MarkerDataset(test_examples,tokenizer);loader=DataLoader(train_set,batch_size=16,shuffle=True);test_loader=DataLoader(test_set,batch_size=32);device='cuda';counts=np.bincount(np.asarray(train_set.labels),minlength=2);history=[]
 if a.reuse_marker_model:
  if not a.marker_model_dir or not (a.marker_model_dir/'config.json').exists():raise FileNotFoundError('A saved --marker-model-dir is required with --reuse-marker-model')
  tokenizer=AutoTokenizer.from_pretrained(a.marker_model_dir);test_set=MarkerDataset(test_examples,tokenizer);test_loader=DataLoader(test_set,batch_size=32);model=AutoModelForSequenceClassification.from_pretrained(a.marker_model_dir).to(device)
 else:
  model=AutoModelForSequenceClassification.from_pretrained(model_name,num_labels=2).to(device);weights=torch.tensor([len(train_set)/(2*max(1,x)) for x in counts],device=device,dtype=torch.float);optimizer=torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay=.01);steps=len(loader)*3;scheduler=get_linear_schedule_with_warmup(optimizer,max(1,steps//10),steps)
  for epoch in range(1,4):
   model.train();losses=[]
   for batch in loader:
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(input_ids=batch['input_ids'].to(device),attention_mask=batch['attention_mask'].to(device)).logits;loss=torch.nn.functional.cross_entropy(logits,batch['label'].to(device),weight=weights)
    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step();scheduler.step();losses.append(float(loss.detach()))
   history.append({'epoch':epoch,'loss':statistics.mean(losses)});print(history[-1],flush=True)
  if a.marker_model_dir:model.save_pretrained(a.marker_model_dir);tokenizer.save_pretrained(a.marker_model_dir)
 model.eval();prob=[]
 with torch.inference_mode():
  for batch in test_loader:prob.extend(model(input_ids=batch['input_ids'].to(device),attention_mask=batch['attention_mask'].to(device)).logits.softmax(-1)[:,1].float().cpu().tolist())
 selected=[{'pmc_id':e[2],'offset':e[3],'end':e[4]} for e,z in zip(test_examples,prob) if z>=a.marker_threshold];del model;torch.cuda.empty_cache();ret=LexicalHPORetriever(ontology);sap_name='cambridgeltl/SapBERT-from-PubMedBERT-fulltext';sap_tok=AutoTokenizer.from_pretrained(sap_name);sap=AutoModel.from_pretrained(sap_name).to(device).eval()
 def surface(doc,row):
  for passage in doc['full_text']:
   left=row['offset']-passage['offset'];right=row['end']-passage['offset']
   if 0<=left<right<=len(passage['text']):return passage['text'][left:right]
  return ''
 def enc(texts,batch=256):
  out=[]
  with torch.inference_mode():
   for i in range(0,len(texts),batch):
    x=sap_tok(texts[i:i+batch],padding=True,truncation=True,max_length=32,return_tensors='pt').to(device);out.append(torch.nn.functional.normalize(sap(**x).last_hidden_state[:,0].float(),p=2,dim=1).cpu())
  return torch.cat(out)
 by={str(d['pmc_id']):d for d in test};by_selected=defaultdict(list);surfaces=set()
 for row in selected:row['surface']=surface(by[str(row['pmc_id'])],row);by_selected[str(row['pmc_id'])].append(row);surfaces.add(ret.normalize(row['surface']))
 phrases=ret.phrases;pv=enc(phrases).to(device);surface_list=sorted(surfaces);mapping={}
 if surface_list:
  q=enc(surface_list);scores,indices=(q.to(device)@pv.T).max(1);mapping={s:(sorted(ret.phrase_to_ids[phrases[i]])[0],float(z)) for s,i,z in zip(surface_list,indices.cpu().tolist(),scores.cpu().tolist())}
 predictions=[];added=0
 for doc in test:
  entities=lexical.extract_document(doc,overlap_policy='longest',detect_negation=False);occupied={(e['offset'],e['length']) for e in entities}
  for row in by_selected[str(doc['pmc_id'])]:
   identifier,score=mapping[ret.normalize(row['surface'])];key=(row['offset'],row['end']-row['offset'])
   if score<a.sapbert_threshold or key in occupied:continue
   entities.append({'identifier':identifier,'type':'Phenotype','offset':key[0],'length':key[1],'text':row['surface'],'note':None});occupied.add(key);added+=1
  entities.sort(key=lambda e:(e['offset'],e['length'],e['identifier']));predictions.append({'pmc_id':doc['pmc_id'],'pmid':doc.get('pmid'),'entities':entities,'association':[{'patient_id':p['patient_id'],'phenotype':[]} for p in doc['patient']]})
 issues=validate_submission(test,predictions);raise_for_errors(issues);save_jsonl(predictions,str(a.output_file));report={'scope':'all labelled training docs -> supplied input','input':str(a.input_file),'marker_threshold':a.marker_threshold,'sapbert_threshold':a.sapbert_threshold,'seed':a.seed,'train_candidates':len(train_examples),'test_candidates':len(test_examples),'class_counts':counts.tolist(),'history':history,'selected_before_sapbert':len(selected),'learned_additions':added,'entities':sum(len(d['entities']) for d in predictions),'validation_warnings':[x.to_dict() for x in issues],'elapsed_seconds':time.time()-started,'output':str(a.output_file)};a.report_file.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
