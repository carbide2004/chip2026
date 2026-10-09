#!/usr/bin/env python3
"""Train the frozen BIO entity model on all 80 labelled documents."""
from collections import Counter
import json,math,random,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np,torch
from torch.utils.data import DataLoader
from transformers import AutoModelForTokenClassification,AutoTokenizer,get_linear_schedule_with_warmup
from src.data_loader import load_jsonl
from train_e3_token_ner import WindowDataset
model_name='microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext';out=ROOT/'data/processed/final/token_ner';out.mkdir(parents=True,exist_ok=True);seed=2026;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);started=time.time();docs=load_jsonl(str(ROOT/'data/raw/PatientPheX-V1-A/PatientPheX-train.jsonl'));tokenizer=AutoTokenizer.from_pretrained(model_name,use_fast=True);dataset=WindowDataset(docs,tokenizer);loader=DataLoader(dataset,batch_size=8,shuffle=True);device='cuda';model=AutoModelForTokenClassification.from_pretrained(model_name,num_labels=3,id2label={0:'O',1:'B',2:'I'},label2id={'O':0,'B':1,'I':2}).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=3e-5,weight_decay=.01);steps=len(loader)*5;scheduler=get_linear_schedule_with_warmup(optimizer,max(1,steps//10),steps);counts=Counter(y for item in dataset.items for y in item['labels'].tolist() if y>=0);total=sum(counts.values());weights=torch.tensor([math.sqrt(total/(3*counts[i])) for i in range(3)],device=device);history=[]
for epoch in range(1,6):
 model.train();losses=[]
 for batch in loader:
  optimizer.zero_grad(set_to_none=True);ids=batch['input_ids'].to(device);mask=batch['attention_mask'].to(device);labels=batch['labels'].to(device)
  with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(input_ids=ids,attention_mask=mask).logits;loss=torch.nn.functional.cross_entropy(logits.view(-1,3),labels.view(-1),weight=weights,ignore_index=-100)
  loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step();scheduler.step();losses.append(loss.detach().item())
 row={'epoch':epoch,'train_loss':sum(losses)/len(losses)};history.append(row);print(json.dumps(row),flush=True)
model.save_pretrained(out/'model');tokenizer.save_pretrained(out/'model');report={'scope':'all 80 labelled documents; frozen five-epoch configuration','model':model_name,'revision':getattr(model.config,'_commit_hash',None),'documents':len(docs),'windows':len(dataset),'label_counts':dict(counts),'class_weights':weights.cpu().tolist(),'dropped_overlap_span_projections':dataset.dropped_overlap_spans,'history':history,'elapsed_seconds':time.time()-started};(out/'training_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False,indent=2))
