#!/usr/bin/env python3
"""PL-Marker-inspired binary classifier for low-threshold BIO span candidates."""
from collections import defaultdict
import argparse,json,random,re,statistics,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np,torch
from torch.utils.data import Dataset,DataLoader
from transformers import AutoModelForSequenceClassification,AutoModelForTokenClassification,AutoTokenizer,get_linear_schedule_with_warmup
from src.baseline_v1 import LexicalPhenotypeBaseline
from src.data_loader import HPOntology,load_jsonl,save_jsonl
from train_e3_token_ner import WindowDataset,decode_batch
class MarkerDataset(Dataset):
 def __init__(self,examples,tokenizer,max_length=256):
  texts=[x[0] for x in examples];self.labels=[x[1] for x in examples];self.enc=tokenizer(texts,padding='max_length',truncation=True,max_length=max_length)
 def __len__(self):return len(self.labels)
 def __getitem__(self,i):return {'input_ids':torch.tensor(self.enc['input_ids'][i]),'attention_mask':torch.tensor(self.enc['attention_mask'][i]),'label':torch.tensor(self.labels[i])}
def bio_spans(docs,model_dir,threshold=.30,batch_size=8):
 tokenizer=AutoTokenizer.from_pretrained(model_dir);dataset=WindowDataset(docs,tokenizer,include_metadata=True);loader=DataLoader(dataset,batch_size=batch_size);device='cuda';model=AutoModelForTokenClassification.from_pretrained(model_dir).to(device).eval();pred=defaultdict(set)
 with torch.inference_mode():
  for batch in loader:
   logits=model(input_ids=batch['input_ids'].to(device),attention_mask=batch['attention_mask'].to(device)).logits.float().cpu();probs=logits.softmax(-1);entity_prob=1-probs[:,:,0];entity_label=probs[:,:,1:].argmax(-1)+1;labels=torch.where(entity_prob>=threshold,entity_label,torch.zeros_like(entity_label));decode_batch(labels,batch,pred)
 del model;torch.cuda.empty_cache();return pred
def marker_text(doc,left,right):
 for passage in doc['full_text']:
  base=passage['offset'];a=left-base;b=right-base
  if 0<=a<b<=len(passage['text']):
   lo=max(0,a-420);hi=min(len(passage['text']),b+420);text=passage['text'];return text[lo:a]+' [unused1] '+text[a:b]+' [unused2] '+text[b:hi]
 return ''
def add_flexible_lexical_spans(docs,bio,ontology):
 token_re=re.compile(r'[a-z0-9]+');index=defaultdict(list)
 for term in ontology.terms.values():
  if term.is_obsolete or not term.is_sub_abnormality:continue
  for phrase in [term.name,*term.synonyms]:
   sequence=tuple(token_re.findall(phrase.casefold()))
   if sequence:index[sequence[0]].append(sequence)
 for values in index.values():values.sort(key=lambda x:-len(x))
 for doc in docs:
  pmc=str(doc['pmc_id'])
  for passage in doc['full_text']:
   tokens=list(token_re.finditer(passage['text'].casefold()))
   for i,token in enumerate(tokens):
    for sequence in index.get(token.group(),[]):
     if i+len(sequence)<=len(tokens) and tuple(x.group() for x in tokens[i:i+len(sequence)])==sequence:
      bio[pmc].add((passage['offset']+tokens[i].start(),passage['offset']+tokens[i+len(sequence)-1].end()));break
 return bio

def candidates(docs,bio,lexical):
 output={};examples=[]
 for doc in docs:
  pmc=str(doc['pmc_id']);lex=lexical.extract_document(doc,overlap_policy='longest',detect_negation=False);lex_spans={(e['offset'],e['offset']+e['length']) for e in lex};gold={(e['offset'],e['offset']+e['length']) for e in doc['entities'] if not e.get('negated',False)};extra=sorted(set(bio.get(pmc,set()))-lex_spans);output[pmc]={'lexical':lex,'extra':extra,'gold':gold}
  examples.extend((marker_text(doc,a,b),int((a,b) in gold),pmc,a,b) for a,b in extra)
 return output,examples
def span_metric(rows,accepted):
 tp=fp=fn=0
 for pmc,row in rows.items():pred={(e['offset'],e['offset']+e['length']) for e in row['lexical']}|set(accepted.get(pmc,set()));gold=row['gold'];tp+=len(pred&gold);fp+=len(pred-gold);fn+=len(gold-pred)
 p=tp/(tp+fp) if tp+fp else 0;r=tp/(tp+fn) if tp+fn else 0;return {'precision':p,'recall':r,'f1':2*p*r/(p+r) if p+r else 0,'tp':tp,'fp':fp,'fn':fn}
def main():
 p=argparse.ArgumentParser();p.add_argument('--fold',type=int,default=0);p.add_argument('--epochs',type=int,default=3);p.add_argument('--seed',type=int,default=2026);p.add_argument('--flexible-lexical',action='store_true');p.add_argument('--model',default='microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext');p.add_argument('--bio-model-dir',type=Path,default=ROOT/'data/processed/e3/token_ner_fold0/model');p.add_argument('--output-dir',type=Path,default=ROOT/'data/processed/e14/marker_fold0');a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True);random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed);raw=ROOT/'data/raw/PatientPheX-V1-A';docs=load_jsonl(str(raw/'PatientPheX-train.jsonl'));by={str(d['pmc_id']):d for d in docs};split=json.loads((ROOT/'data/processed/e0/splits.json').read_text())['cv_folds'][a.fold];train=[by[x] for x in split['train']];val=[by[x] for x in split['validation']];ontology=HPOntology(str(raw/'hp.obo'));lexical=LexicalPhenotypeBaseline(ontology);train_bio=bio_spans(train,a.bio_model_dir);val_bio=bio_spans(val,a.bio_model_dir)
 if a.flexible_lexical:train_bio=add_flexible_lexical_spans(train,train_bio,ontology);val_bio=add_flexible_lexical_spans(val,val_bio,ontology)
 train_rows,train_examples=candidates(train,train_bio,lexical);val_rows,val_examples=candidates(val,val_bio,lexical);tokenizer=AutoTokenizer.from_pretrained(a.model);train_set=MarkerDataset(train_examples,tokenizer);val_set=MarkerDataset(val_examples,tokenizer);loader=DataLoader(train_set,batch_size=16,shuffle=True);vloader=DataLoader(val_set,batch_size=32);device='cuda';model=AutoModelForSequenceClassification.from_pretrained(a.model,num_labels=2).to(device);counts=np.bincount(np.asarray(train_set.labels),minlength=2);weights=torch.tensor([len(train_set)/(2*max(1,x)) for x in counts],device=device,dtype=torch.float);optimizer=torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay=.01);steps=len(loader)*a.epochs;scheduler=get_linear_schedule_with_warmup(optimizer,max(1,steps//10),steps);history=[];started=time.time()
 for epoch in range(1,a.epochs+1):
  model.train();losses=[]
  for batch in loader:
   optimizer.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(input_ids=batch['input_ids'].to(device),attention_mask=batch['attention_mask'].to(device)).logits;loss=torch.nn.functional.cross_entropy(logits,batch['label'].to(device),weight=weights)
   loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step();scheduler.step();losses.append(float(loss.detach()))
  history.append({'epoch':epoch,'loss':statistics.mean(losses)});print(history[-1],flush=True)
 model.eval();prob=[]
 with torch.inference_mode():
  for batch in vloader:prob.extend(model(input_ids=batch['input_ids'].to(device),attention_mask=batch['attention_mask'].to(device)).logits.softmax(-1)[:,1].float().cpu().tolist())
 with (a.output_dir/'candidate_probabilities.jsonl').open('w',encoding='utf-8') as handle:
  for example,z in zip(val_examples,prob):handle.write(json.dumps({'pmc_id':example[2],'offset':example[3],'end':example[4],'text':example[0],'gold_label':example[1],'probability':float(z)},ensure_ascii=False)+'\n')
 runs=[]
 for threshold in [.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9]:
  accepted=defaultdict(set)
  for example,z in zip(val_examples,prob):
   if z>=threshold:accepted[example[2]].add((example[3],example[4]))
  runs.append({'threshold':threshold,**span_metric(val_rows,accepted)});print(runs[-1])
 best=max(runs,key=lambda x:x['f1']);accepted=defaultdict(set)
 for example,z in zip(val_examples,prob):
  if z>=best['threshold']:accepted[example[2]].add((example[3],example[4]))
 output=[]
 for doc in val:
  pmc=str(doc['pmc_id']);entities=list(val_rows[pmc]['lexical'])
  for left,right in sorted(accepted[pmc]):
   for passage in doc['full_text']:
    x=left-passage['offset'];y=right-passage['offset']
    if 0<=x<y<=len(passage['text']):entities.append({'identifier':'-1','type':'Phenotype','offset':left,'length':right-left,'text':passage['text'][x:y],'note':None});break
  output.append({'pmc_id':doc['pmc_id'],'pmid':doc.get('pmid'),'entities':entities,'association':[]})
 save_jsonl(output,str(a.output_dir/'span_predictions.jsonl'));report={'scope':f'CV fold {a.fold}; locked validation untouched','train_candidates':len(train_examples),'validation_candidates':len(val_examples),'class_counts':counts.tolist(),'candidate_ceiling':span_metric(val_rows,{k:set(v['extra']) for k,v in val_rows.items()}),'history':history,'runs':runs,'best':best,'elapsed_seconds':time.time()-started};(a.output_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
