#!/usr/bin/env python3
"""Train frozen E12 relation text model on all 80 docs and score A candidates."""
import argparse,json,random,statistics,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np,torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification,AutoTokenizer,get_linear_schedule_with_warmup
from src.data_loader import HPOntology,load_jsonl
from src.relation_features import build_examples
from train_e12_relation_text import PairDataset,pair_text
p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--model',default='microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext');p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True);random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed);torch.use_deterministic_algorithms(True,warn_only=True);torch.backends.cudnn.benchmark=False;raw=ROOT/'data/raw/PatientPheX-V1-A';train=load_jsonl(str(raw/'PatientPheX-train.jsonl'));test_source=load_jsonl(str(raw/'PatientPheX-A.jsonl'));entity_rows={str(d['pmc_id']):d['entities'] for d in load_jsonl(str(ROOT/'data/processed/final/a_gated_entities.jsonl'))};test=[]
for d in test_source:item=dict(d);item['entities']=entity_rows[str(d['pmc_id'])];test.append(item)
_,labels,train_keys=build_examples(train);_,_,test_keys=build_examples(test);ontology=HPOntology(str(raw/'hp.obo'));train_by={str(d['pmc_id']):d for d in train};test_by={str(d['pmc_id']):d for d in test};train_text=[pair_text(train_by[k[0]],k[1],k[2],ontology) for k in train_keys];test_text=[pair_text(test_by[k[0]],k[1],k[2],ontology) for k in test_keys];tokenizer=AutoTokenizer.from_pretrained(a.model);train_set=PairDataset(train_text,labels,tokenizer);test_set=PairDataset(test_text,[0]*len(test_keys),tokenizer);loader=DataLoader(train_set,batch_size=8,shuffle=True);test_loader=DataLoader(test_set,batch_size=16);device='cuda';model=AutoModelForSequenceClassification.from_pretrained(a.model,num_labels=2).to(device);counts=np.bincount(np.asarray(labels),minlength=2);weights=torch.tensor([len(labels)/(2*max(1,x)) for x in counts],device=device,dtype=torch.float);optimizer=torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay=.01);steps=len(loader)*3;scheduler=get_linear_schedule_with_warmup(optimizer,max(1,steps//10),steps);history=[];started=time.time()
for epoch in range(1,4):
 model.train();losses=[]
 for batch in loader:
  optimizer.zero_grad(set_to_none=True)
  with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(input_ids=batch['input_ids'].to(device),attention_mask=batch['attention_mask'].to(device)).logits;loss=torch.nn.functional.cross_entropy(logits,batch['labels'].to(device),weight=weights)
  loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step();scheduler.step();losses.append(float(loss.detach()))
 history.append({'epoch':epoch,'loss':statistics.mean(losses)});print(history[-1],flush=True)
model.eval();prob=[]
with torch.inference_mode():
 for batch in test_loader:prob.extend(model(input_ids=batch['input_ids'].to(device),attention_mask=batch['attention_mask'].to(device)).logits.softmax(-1)[:,1].float().cpu().tolist())
with (a.output_dir/'a_probabilities.jsonl').open('w',encoding='utf-8') as handle:
 for key,prediction in zip(test_keys,prob):handle.write(json.dumps({'key':list(key),'probability':float(prediction)},ensure_ascii=False)+'\n')
report={'scope':'all 80 labelled docs -> A candidates','seed':a.seed,'train_pairs':len(train_set),'test_pairs':len(test_set),'class_counts':counts.tolist(),'history':history,'elapsed_seconds':time.time()-started};(a.output_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
