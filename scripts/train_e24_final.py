#!/usr/bin/env python3
"""Train E24 HPO contextual reranker on all 80 labels and rerank A entities."""
from copy import deepcopy
from pathlib import Path
import argparse,json,random,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import numpy as np,torch
from torch.utils.data import DataLoader
from transformers import AutoModel,AutoModelForSequenceClassification,AutoTokenizer,get_linear_schedule_with_warmup
from src.data_loader import HPOntology,load_jsonl,save_jsonl
from src.hpo_retrieval import LexicalHPORetriever
from run_e2_dense_retrieval import encode
from train_e24_hpo_reranker import records,context,candidate_text,PairDataset

def main():
 p=argparse.ArgumentParser();p.add_argument('--entities-file',type=Path,default=ROOT/'data/processed/final/a_e14_entities.jsonl');p.add_argument('--output-file',type=Path,default=ROOT/'data/processed/final/a_e24_entities.jsonl');p.add_argument('--report-file',type=Path,default=ROOT/'data/processed/final/e24_report.json');p.add_argument('--seed',type=int,default=2026);p.add_argument('--epochs',type=int,default=2);p.add_argument('--alpha',type=float,default=16);p.add_argument('--encoder',default='microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext');a=p.parse_args();random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed);started=time.time();raw=ROOT/'data/raw/PatientPheX-V1-A';train_docs=load_jsonl(str(raw/'PatientPheX-train.jsonl'));test_docs=load_jsonl(str(raw/'PatientPheX-A.jsonl'));by_test={str(x['pmc_id']):x for x in test_docs};predicted_rows=load_jsonl(str(a.entities_file));ontology=HPOntology(str(raw/'hp.obo'));retriever=LexicalHPORetriever(ontology);train=records(train_docs,ontology,retriever);surfaces=sorted({x['surface'] for x in train}|{retriever.normalize(e.get('text','')) for d in predicted_rows for e in d.get('entities',[])});device='cuda';sap_name='cambridgeltl/SapBERT-from-PubMedBERT-fulltext';sap_tok=AutoTokenizer.from_pretrained(sap_name);sap=AutoModel.from_pretrained(sap_name).to(device);phrases=retriever.phrases;phrase_ids=[sorted(retriever.phrase_to_ids[x]) for x in phrases];pv=encode(phrases,sap_tok,sap,device).to(device);qv=encode(surfaces,sap_tok,sap,device);rankings={}
 with torch.inference_mode():
  for start in range(0,len(surfaces),128):
   values,indices=(qv[start:start+128].to(device)@pv.T).topk(300,dim=1)
   for surface,vs,ix in zip(surfaces[start:start+128],values.cpu().tolist(),indices.cpu().tolist()):
    scores={}
    for score,j in zip(vs,ix):
     for hid in phrase_ids[j]:scores[hid]=max(scores.get(hid,-1),score)
    rankings[surface]=sorted(scores.items(),key=lambda x:(-x[1],x[0]))[:20]
 del sap,pv,qv;torch.cuda.empty_cache();pairs=[];covered=0
 for r in train:
  candidates=[x[0] for x in rankings[r['surface']]]
  if r['identifier'] not in candidates:continue
  covered+=1;pairs.append((r['context'],candidate_text(r['identifier'],ontology),1));negatives=[x for x in candidates if x!=r['identifier']][:5];pairs.extend((r['context'],candidate_text(x,ontology),0) for x in negatives)
 tokenizer=AutoTokenizer.from_pretrained(a.encoder);dataset=PairDataset(pairs,tokenizer);loader=DataLoader(dataset,batch_size=16,shuffle=True,collate_fn=lambda rows:tokenizer.pad(rows,padding=True,return_tensors='pt'));model=AutoModelForSequenceClassification.from_pretrained(a.encoder,num_labels=2).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay=.01);steps=len(loader)*a.epochs;scheduler=get_linear_schedule_with_warmup(optimizer,max(1,steps//10),steps);history=[]
 for epoch in range(1,a.epochs+1):
  model.train();losses=[]
  for batch in loader:
   batch={k:v.to(device) for k,v in batch.items()};optimizer.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.bfloat16):loss=model(**batch).loss
   loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step();scheduler.step();losses.append(float(loss.detach()))
  history.append({'epoch':epoch,'loss':sum(losses)/len(losses)});print(history[-1],flush=True)
 model.eval();output=[];changed=[]
 with torch.inference_mode():
  for predicted in predicted_rows:
   source=by_test[str(predicted['pmc_id'])];row=deepcopy(predicted);entities=[]
   for entity in row.get('entities',[]):
    item=deepcopy(entity);surface=retriever.normalize(item.get('text',''));current=ontology.resolve_id(item.get('identifier')) or item.get('identifier');exact=retriever.phrase_to_ids.get(surface,set())
    if current not in exact:
     ranked=rankings[surface];ctx=context(source,item);texts=[candidate_text(x[0],ontology) for x in ranked];logits=[]
     for start in range(0,len(ranked),64):
      x=tokenizer([ctx]*len(ranked[start:start+64]),texts[start:start+64],padding=True,truncation=True,max_length=256,return_tensors='pt').to(device);logits.extend(model(**x).logits[:,1].float().cpu().tolist())
     choice=max(range(len(ranked)),key=lambda i:logits[i]+a.alpha*ranked[i][1]);new=ranked[choice][0]
     if new!=current:changed.append({'pmc_id':str(predicted['pmc_id']),'offset':item['offset'],'text':item.get('text'),'old':current,'new':new});item['identifier']=new
    entities.append(item)
   row['entities']=entities;output.append(row)
 a.output_file.parent.mkdir(parents=True,exist_ok=True);save_jsonl(output,str(a.output_file));report={'scope':'all 80 labelled docs -> A; E24 config frozen on development OOF','alpha':a.alpha,'epochs':a.epochs,'seed':a.seed,'train_records':len(train),'covered_train':covered,'train_pairs':len(pairs),'documents':len(output),'entities':sum(len(x['entities']) for x in output),'changed_count':len(changed),'changes':changed,'history':history,'elapsed_seconds':time.time()-started,'input':str(a.entities_file),'output':str(a.output_file)};a.report_file.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='changes'},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
