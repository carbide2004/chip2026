#!/usr/bin/env python3
"""EIDER-inspired occurrence classifier trained from high-confidence silver evidence."""
from collections import defaultdict
from copy import deepcopy
import argparse,json,random,statistics,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np,torch
from torch.utils.data import Dataset,DataLoader
from transformers import AutoModelForSequenceClassification,AutoTokenizer,get_linear_schedule_with_warmup
from src.data_loader import HPOntology,load_jsonl
from src.metrics import compute_all_metrics
from src.relation_features import association_set,build_examples,passage_index,positive_occurrences
from run_e6_llm_router import passage_for_offset,relation_prediction
class TextDataset(Dataset):
 def __init__(self,examples,tokenizer,max_length=256):
  self.examples=examples;self.enc=tokenizer([x['text'] for x in examples],padding='max_length',truncation=True,max_length=max_length)
 def __len__(self):return len(self.examples)
 def __getitem__(self,i):return {'input_ids':torch.tensor(self.enc['input_ids'][i]),'attention_mask':torch.tensor(self.enc['attention_mask'][i]),'label':torch.tensor(self.examples[i].get('label',0))}
def occurrence_text(doc,patient_id,value,occurrence,ontology):
 patient=next(p for p in doc['patient'] if str(p['patient_id'])==patient_id);aliases=', '.join(f'{m["text"]}@{m["offset"]}' for m in patient.get('mention',[])) or 'none';others=[]
 for p in doc['patient']:
  if str(p['patient_id'])!=patient_id:others.append(f'{p["patient_id"]}='+','.join(m['text'] for m in p.get('mention',[])[:3]))
 passage=doc['full_text'][occurrence['passage_index']] if occurrence['passage_index'] is not None else passage_for_offset(doc,occurrence['offset']);local=occurrence['offset']-passage['offset'];left=max(0,local-320);right=min(len(passage['text']),local+320);snippet=' '.join(passage['text'][left:right].split());term=ontology.terms.get(value);return f'TARGET {patient_id}; aliases {aliases}. OTHER PATIENTS {"; ".join(others) or "none"}. PHENOTYPE {value} {term.name if term else ""}. SECTION {passage.get("section_type")}. OCCURRENCE CONTEXT {snippet}'
def anchor_owners(doc,occurrence,radius=500):
 same=set();nearest=None;best=10**18
 for p in doc['patient']:
  owner=str(p['patient_id'])
  for mention in p.get('mention',[]):
   pidx=passage_index(doc,mention['offset']);distance=abs(mention['offset']-occurrence['offset'])
   if pidx==occurrence['passage_index']:same.add(owner)
   if distance<best:best=distance;nearest=owner
 return same,nearest,best
def silver_examples(docs,ontology,negative_ratio=3):
 positive=[];negative=[]
 for doc in docs:
  occurrences=positive_occurrences(doc);gold={str(a['patient_id']):association_set(a) for a in doc.get('association',[])}
  for patient in doc['patient']:
   pid=str(patient['patient_id'])
   for value,items in occurrences.items():
    bag_positive=value in gold.get(pid,set())
    for occurrence in items:
     owners,nearest,distance=anchor_owners(doc,occurrence);label=None
     if bag_positive:
      if owners=={pid} or (not owners and nearest==pid and distance<=500):label=1
      elif owners and pid not in owners:label=0
     else:label=0
     if label is None:continue
     row={'text':occurrence_text(doc,pid,value,occurrence,ontology),'label':label}
     (positive if label else negative).append(row)
 random.Random(2026).shuffle(negative);negative=negative[:max(len(positive)*negative_ratio,len(positive))];rows=positive+negative;random.Random(2026).shuffle(rows);return rows,{'positive':len(positive),'negative':len(negative)}
def inference_examples(docs,keys,ontology):
 by={str(d['pmc_id']):d for d in docs};rows=[];ranges=[]
 for key in keys:
  doc=by[key[0]];items=positive_occurrences(doc).get(key[2],[]);start=len(rows)
  for occurrence in items:rows.append({'text':occurrence_text(doc,key[1],key[2],occurrence,ontology),'key':key})
  ranges.append((start,len(rows)))
 return rows,ranges
def main():
 p=argparse.ArgumentParser();p.add_argument('--fold',type=int,default=0);p.add_argument('--seed',type=int,default=2026);p.add_argument('--epochs',type=int,default=3);p.add_argument('--model',default='microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext');p.add_argument('--output-dir',type=Path,default=ROOT/'data/processed/e16/fold0');a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True);random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed);raw=ROOT/'data/raw/PatientPheX-V1-A';docs=load_jsonl(str(raw/'PatientPheX-train.jsonl'));by={str(d['pmc_id']):d for d in docs};split=json.loads((ROOT/'data/processed/e0/splits.json').read_text())['cv_folds'][a.fold];train=[by[x] for x in split['train']];gold=[by[x] for x in split['validation']];sup={str(d['pmc_id']):d['entities'] for d in load_jsonl(str(ROOT/'data/processed/e14/gated_oof'/f'fold{a.fold}.jsonl'))};validation=[]
 for doc in gold:item=deepcopy(doc);item['entities']=sup[str(doc['pmc_id'])];validation.append(item)
 _,_,keys=build_examples(validation);ontology=HPOntology(str(raw/'hp.obo'));train_examples,class_counts=silver_examples(train,ontology);inference_rows,ranges=inference_examples(validation,keys,ontology);tokenizer=AutoTokenizer.from_pretrained(a.model);train_set=TextDataset(train_examples,tokenizer);test_set=TextDataset(inference_rows,tokenizer);loader=DataLoader(train_set,batch_size=16,shuffle=True);test_loader=DataLoader(test_set,batch_size=32);device='cuda';model=AutoModelForSequenceClassification.from_pretrained(a.model,num_labels=2).to(device);weights=torch.tensor([len(train_set)/(2*class_counts['negative']),len(train_set)/(2*class_counts['positive'])],device=device);optimizer=torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay=.01);steps=len(loader)*a.epochs;scheduler=get_linear_schedule_with_warmup(optimizer,max(1,steps//10),steps);history=[];started=time.time()
 for epoch in range(1,a.epochs+1):
  model.train();losses=[]
  for batch in loader:
   optimizer.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(input_ids=batch['input_ids'].to(device),attention_mask=batch['attention_mask'].to(device)).logits;loss=torch.nn.functional.cross_entropy(logits,batch['label'].to(device),weight=weights)
   loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step();scheduler.step();losses.append(float(loss.detach()))
  history.append({'epoch':epoch,'loss':statistics.mean(losses)});print(history[-1],flush=True)
 model.eval();instance_prob=[]
 with torch.inference_mode():
  for batch in test_loader:instance_prob.extend(model(input_ids=batch['input_ids'].to(device),attention_mask=batch['attention_mask'].to(device)).logits.softmax(-1)[:,1].float().cpu().tolist())
 bag_prob=[max(instance_prob[start:end],default=0) for start,end in ranges];entities={str(d['pmc_id']):d['entities'] for d in validation};runs=[]
 for threshold in [.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95]:
  pred=relation_prediction(validation,entities,keys,bag_prob,threshold);m=compute_all_metrics(gold,pred);runs.append({'threshold':threshold,'metrics':m});print(threshold,m['overall_score'],m['subtask2']['f1_micro'],m['subtask2']['f1_macro'])
 with (a.output_dir/'validation_probabilities.jsonl').open('w') as handle:
  for key,z in zip(keys,bag_prob):handle.write(json.dumps({'key':list(key),'probability':z})+'\n')
 best=max(runs,key=lambda x:x['metrics']['overall_score']);report={'scope':f'CV fold {a.fold}; locked validation untouched','method':'silver occurrence evidence + max MIL','silver_counts':class_counts,'train_instances':len(train_set),'validation_instances':len(test_set),'history':history,'runs':runs,'best':best,'elapsed_seconds':time.time()-started};(a.output_dir/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(best,indent=2))
if __name__=='__main__':main()
