#!/usr/bin/env python3
"""E24 contextual cross-encoder reranking of SapBERT HPO candidates."""
from __future__ import annotations
from pathlib import Path
from collections import defaultdict
from copy import deepcopy
import argparse,json,random,statistics,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np,torch
from torch.utils.data import Dataset,DataLoader
from transformers import AutoModel,AutoModelForSequenceClassification,AutoTokenizer,get_linear_schedule_with_warmup
from src.data_loader import HPOntology,load_jsonl,save_jsonl
from src.hpo_retrieval import LexicalHPORetriever
from src.metrics import compute_all_metrics
from run_e2_dense_retrieval import encode

def split_ids(raw):return [x.strip() for x in str(raw).split(';') if x.strip()]
def context(doc,entity):
 for p in doc['full_text']:
  local=int(entity['offset'])-int(p['offset']);length=int(entity.get('length',len(entity.get('text',''))))
  if 0<=local< len(p['text']):
   a=max(0,local-220);b=min(len(p['text']),local+length+220);text=p['text'];return ' '.join((text[a:local]+' [unused1] '+text[local:local+length]+' [unused2] '+text[local+length:b]).split())
 return entity.get('text','')
def candidate_text(identifier,ontology):
 t=ontology.terms[identifier];parents='; '.join(ontology.terms[x].name for x in t.is_a[:3] if x in ontology.terms);syn='; '.join(t.synonyms[:6]);return f'{t.name}. Synonyms: {syn or "none"}. Definition: {t.definition or "none"}. Parents: {parents or "none"}.'
def records(docs,ontology,retriever):
 out=[]
 for doc in docs:
  for e in doc.get('entities',[]):
   if str(e.get('note') or '').upper()=='NO':continue
   ids=[]
   for raw in split_ids(e.get('identifier')):
    resolved=ontology.resolve_id(raw)
    if resolved and ontology.terms[resolved].is_sub_abnormality:ids.append(resolved)
   for identifier in ids:out.append({'pmc_id':str(doc['pmc_id']),'surface':retriever.normalize(e.get('text','')),'context':context(doc,e),'identifier':identifier})
 return out
class PairDataset(Dataset):
 def __init__(self,pairs,tokenizer,max_length=256):self.pairs=pairs;self.tokenizer=tokenizer;self.max_length=max_length
 def __len__(self):return len(self.pairs)
 def __getitem__(self,i):
  a,b,y=self.pairs[i];x=self.tokenizer(a,b,truncation=True,max_length=self.max_length);x['labels']=y;return x
def main():
 p=argparse.ArgumentParser();p.add_argument('--fold',type=int,default=0);p.add_argument('--epochs',type=int,default=2);p.add_argument('--seed',type=int,default=2026);p.add_argument('--encoder',default='microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext');p.add_argument('--output-dir',type=Path,default=ROOT/'data/processed/e24/fold0');a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True);random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed);started=time.time();raw=ROOT/'data/raw/PatientPheX-V1-A';docs=load_jsonl(str(raw/'PatientPheX-train.jsonl'));by={str(x['pmc_id']):x for x in docs};split=json.loads((ROOT/'data/processed/e0/splits.json').read_text())['cv_folds'][a.fold];ontology=HPOntology(str(raw/'hp.obo'));retriever=LexicalHPORetriever(ontology);train=records([by[x] for x in split['train']],ontology,retriever);validation=records([by[x] for x in split['validation']],ontology,retriever);predicted_rows=load_jsonl(str(ROOT/f'data/processed/e14/gated_oof/fold{a.fold}.jsonl'));predicted_surfaces={retriever.normalize(e.get('text','')) for d in predicted_rows for e in d.get('entities',[])};surfaces=sorted({x['surface'] for x in train+validation}|predicted_surfaces);device='cuda';sap_name='cambridgeltl/SapBERT-from-PubMedBERT-fulltext';sap_tok=AutoTokenizer.from_pretrained(sap_name);sap=AutoModel.from_pretrained(sap_name).to(device);phrases=retriever.phrases;phrase_ids=[sorted(retriever.phrase_to_ids[x]) for x in phrases];pv=encode(phrases,sap_tok,sap,device).to(device);qv=encode(surfaces,sap_tok,sap,device);rankings={}
 with torch.inference_mode():
  for start in range(0,len(surfaces),128):
   values,indices=(qv[start:start+128].to(device)@pv.T).topk(300,dim=1)
   for surface,vs,ix in zip(surfaces[start:start+128],values.cpu().tolist(),indices.cpu().tolist()):
    scores={}
    for score,j in zip(vs,ix):
     for hid in phrase_ids[j]:scores[hid]=max(scores.get(hid,-1),score)
    rankings[surface]=sorted(scores.items(),key=lambda x:(-x[1],x[0]))[:20]
 del sap,pv,qv;torch.cuda.empty_cache();train_pairs=[];covered_train=0
 for r in train:
  candidates=[x[0] for x in rankings[r['surface']]]
  if r['identifier'] not in candidates:continue
  covered_train+=1;train_pairs.append((r['context'],candidate_text(r['identifier'],ontology),1));neg=[x for x in candidates if x!=r['identifier']][:5]
  train_pairs.extend((r['context'],candidate_text(x,ontology),0) for x in neg)
 tokenizer=AutoTokenizer.from_pretrained(a.encoder);model=AutoModelForSequenceClassification.from_pretrained(a.encoder,num_labels=2).to(device);dataset=PairDataset(train_pairs,tokenizer);collator=lambda rows:tokenizer.pad(rows,padding=True,return_tensors='pt');loader=DataLoader(dataset,batch_size=16,shuffle=True,collate_fn=collator);optimizer=torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay=.01);steps=len(loader)*a.epochs;scheduler=get_linear_schedule_with_warmup(optimizer,max(1,steps//10),steps);history=[]
 for epoch in range(1,a.epochs+1):
  model.train();losses=[]
  for batch in loader:
   batch={k:v.to(device) for k,v in batch.items()};optimizer.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.bfloat16):loss=model(**batch).loss
   loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step();scheduler.step();losses.append(float(loss.detach()))
  history.append({'epoch':epoch,'loss':statistics.mean(losses)});print(history[-1],flush=True)
 model.eval();all_scores=[]
 with torch.inference_mode():
  for r in validation:
   ranked=rankings[r['surface']];pairs=[(r['context'],candidate_text(x[0],ontology)) for x in ranked];logits=[]
   for start in range(0,len(pairs),64):
    x=tokenizer([z[0] for z in pairs[start:start+64]],[z[1] for z in pairs[start:start+64]],padding=True,truncation=True,max_length=256,return_tensors='pt').to(device);logits.extend(model(**x).logits[:,1].float().cpu().tolist())
   all_scores.append((r,ranked,logits))
 alphas=[0,.5,1,2,4,8,16];runs=[]
 for alpha in alphas:
  hits=covered=0
  for r,ranked,logits in all_scores:
   ids=[x[0] for x in ranked];covered+=int(r['identifier'] in ids);best=max(range(len(ids)),key=lambda i:logits[i]+alpha*ranked[i][1]);hits+=int(ids[best]==r['identifier'])
  runs.append({'alpha':alpha,'hits':hits,'accuracy':hits/len(validation),'coverage20':covered/len(validation)})
 baseline=sum(r['identifier']==ranked[0][0] for r,ranked,_ in all_scores)/len(validation);best=max(runs,key=lambda x:x['accuracy']);reranked_rows=[];changed=0
 with torch.inference_mode():
  for predicted in predicted_rows:
   source_doc=by[str(predicted['pmc_id'])];row=deepcopy(predicted);entities=[]
   for entity in row.get('entities',[]):
    item=deepcopy(entity);surface=retriever.normalize(item.get('text',''));current=ontology.resolve_id(item.get('identifier')) or item.get('identifier');exact_ids=retriever.phrase_to_ids.get(surface,set())
    if current not in exact_ids and surface in rankings:
     ranked=rankings[surface];ctx=context(source_doc,item);pairs=[(ctx,candidate_text(x[0],ontology)) for x in ranked];logits=[]
     for start in range(0,len(pairs),64):
      x=tokenizer([z[0] for z in pairs[start:start+64]],[z[1] for z in pairs[start:start+64]],padding=True,truncation=True,max_length=256,return_tensors='pt').to(device);logits.extend(model(**x).logits[:,1].float().cpu().tolist())
     choice=max(range(len(ranked)),key=lambda i:logits[i]+16*ranked[i][1]);new_id=ranked[choice][0]
     if new_id!=current:item['identifier']=new_id;changed+=1
    entities.append(item)
   row['entities']=entities;reranked_rows.append(row)
 save_jsonl(reranked_rows,str(a.output_dir/'reranked_predictions.jsonl'));gold_docs=[by[x] for x in split['validation']];reranked_metrics=compute_all_metrics(gold_docs,reranked_rows)
 report={'scope':f'fold {a.fold} gold positive spans; locked validation untouched','train_records':len(train),'validation_records':len(validation),'covered_train':covered_train,'train_pairs':len(train_pairs),'sapbert_top1':baseline,'history':history,'runs':runs,'best':best,'predicted_entities':{'changed':changed,'mention_f1':reranked_metrics['subtask1']['f1_men'],'document_f1':reranked_metrics['subtask1']['f1_doc']},'elapsed_seconds':time.time()-started};(a.output_dir/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
