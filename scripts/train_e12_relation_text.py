#!/usr/bin/env python3
"""Fine-tune BiomedBERT for patient-concept evidence-bag classification."""
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
from src.relation_features import build_examples,split_value
from run_e6_llm_router import passage_for_offset,relation_prediction
class PairDataset(Dataset):
 def __init__(self,texts,labels,tokenizer,max_length=512):
  self.enc=tokenizer(texts,padding='max_length',truncation=True,max_length=max_length);self.labels=labels
 def __len__(self):return len(self.labels)
 def __getitem__(self,i):return {'input_ids':torch.tensor(self.enc['input_ids'][i]),'attention_mask':torch.tensor(self.enc['attention_mask'][i]),'labels':torch.tensor(self.labels[i])}
def pair_text(doc,patient_id,value,ontology,max_occurrences=6):
 patient=next(p for p in doc['patient'] if str(p['patient_id'])==patient_id);target=patient.get('mention',[]);target_offsets=[m['offset'] for m in target];aliases=', '.join(f'{m["text"]}@{m["offset"]}' for m in target) or 'none';others=[]
 for p in doc['patient']:
  if str(p['patient_id'])!=patient_id:others.append(f'{p["patient_id"]}='+','.join(m['text'] for m in p.get('mention',[])[:4]))
 term=ontology.terms.get(value);header=f'TARGET PATIENT {patient_id}; aliases: {aliases}. OTHER PATIENTS: {"; ".join(others) or "none"}. PHENOTYPE: {value} {term.name if term else ""}.';occ=[]
 for entity in doc.get('entities',[]):
  if value not in split_value(entity.get('identifier'),entity.get('text','')):continue
  passage=passage_for_offset(doc,entity['offset'])
  if not passage:continue
  local=entity['offset']-passage['offset'];left=max(0,local-220);right=min(len(passage['text']),local+entity['length']+220);context=' '.join(passage['text'][left:right].split());distance=min((entity['offset']-x for x in target_offsets),key=abs) if target_offsets else 999999;occ.append((abs(distance),f'[{passage.get("section_type")}; target-distance={distance:+d}] {context}'))
 occ.sort(key=lambda x:x[0]);return header+' EVIDENCE: '+' [SEP] '.join(x[1] for x in occ[:max_occurrences])
def main():
 p=argparse.ArgumentParser();p.add_argument('--fold',type=int,default=0);p.add_argument('--epochs',type=int,default=3);p.add_argument('--seed',type=int,default=2026);p.add_argument('--model',default='microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext');p.add_argument('--output-dir',type=Path,default=ROOT/'data/processed/e12/fold0');a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True);random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed);torch.use_deterministic_algorithms(True,warn_only=True);torch.backends.cudnn.benchmark=False;raw=ROOT/'data/raw/PatientPheX-V1-A';docs=load_jsonl(str(raw/'PatientPheX-train.jsonl'));by={str(d['pmc_id']):d for d in docs};split=json.loads((ROOT/'data/processed/e0/splits.json').read_text())['cv_folds'][a.fold];train=[by[x] for x in split['train']];gold=[by[x] for x in split['validation']];sup={str(d['pmc_id']):d['entities'] for d in load_jsonl(str(ROOT/'data/processed/e3/gated_oof'/f'fold{a.fold}.jsonl'))};validation=[]
 for doc in gold:item=deepcopy(doc);item['entities']=sup[str(doc['pmc_id'])];validation.append(item)
 _,yt,kt=build_examples(train);_,yv,kv=build_examples(validation);ontology=HPOntology(str(raw/'hp.obo'));train_by={str(d['pmc_id']):d for d in train};val_by={str(d['pmc_id']):d for d in validation};train_text=[pair_text(train_by[k[0]],k[1],k[2],ontology) for k in kt];val_text=[pair_text(val_by[k[0]],k[1],k[2],ontology) for k in kv];tok=AutoTokenizer.from_pretrained(a.model);tr=PairDataset(train_text,yt,tok);va=PairDataset(val_text,yv,tok);loader=DataLoader(tr,batch_size=8,shuffle=True);vloader=DataLoader(va,batch_size=16);device='cuda';model=AutoModelForSequenceClassification.from_pretrained(a.model,num_labels=2).to(device);counts=np.bincount(np.asarray(yt),minlength=2);weights=torch.tensor([len(yt)/(2*max(1,x)) for x in counts],device=device,dtype=torch.float);opt=torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay=.01);steps=len(loader)*a.epochs;sched=get_linear_schedule_with_warmup(opt,max(1,steps//10),steps);history=[];started=time.time()
 for epoch in range(1,a.epochs+1):
  model.train();losses=[]
  for batch in loader:
   opt.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(input_ids=batch['input_ids'].to(device),attention_mask=batch['attention_mask'].to(device)).logits;loss=torch.nn.functional.cross_entropy(logits,batch['labels'].to(device),weight=weights)
   loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();sched.step();losses.append(float(loss.detach()))
  history.append({'epoch':epoch,'loss':statistics.mean(losses)});print(history[-1],flush=True)
 model.eval();prob=[]
 with torch.inference_mode():
  for batch in vloader:prob.extend(model(input_ids=batch['input_ids'].to(device),attention_mask=batch['attention_mask'].to(device)).logits.softmax(-1)[:,1].float().cpu().tolist())
 with (a.output_dir/'validation_probabilities.jsonl').open('w',encoding='utf-8') as handle:
  for key,label,prediction in zip(kv,yv,prob):handle.write(json.dumps({'key':list(key),'label':int(label),'probability':float(prediction)},ensure_ascii=False)+'\n')
 entities={str(d['pmc_id']):d['entities'] for d in validation};runs=[]
 for threshold in [.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8]:
  pred=relation_prediction(validation,entities,kv,prob,threshold);m=compute_all_metrics(gold,pred);runs.append({'threshold':threshold,'metrics':m});print(threshold,m['overall_score'],m['subtask2']['f1_micro'],m['subtask2']['f1_macro'])
 best=max(runs,key=lambda x:x['metrics']['overall_score']);report={'scope':f'CV fold {a.fold}; locked validation untouched','train_pairs':len(tr),'validation_pairs':len(va),'class_counts':counts.tolist(),'history':history,'runs':runs,'best':best,'elapsed_seconds':time.time()-started};(a.output_dir/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(best,indent=2))
if __name__=='__main__':main()
