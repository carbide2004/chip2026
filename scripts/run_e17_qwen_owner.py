#!/usr/bin/env python3
"""Assign each phenotype occurrence to patient IDs with constrained Qwen output."""
from collections import defaultdict
from copy import deepcopy
import argparse,json,re,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM,AutoTokenizer
from src.data_loader import HPOntology,load_jsonl
from src.metrics import compute_all_metrics
from src.relation_features import build_examples,split_value
from run_e6_llm_router import load_cache,passage_for_offset,relation_prediction

def prompt(doc,entity,ontology):
 patients=[]
 for p in doc['patient']:
  mentions=', '.join(f'"{m["text"]}"@{m["offset"]}' for m in p.get('mention',[])[:8]) or 'none';patients.append(f'- {p["patient_id"]}: {mentions}')
 passage=passage_for_offset(doc,entity['offset']);local=entity['offset']-passage['offset'];left=max(0,local-550);right=min(len(passage['text']),local+entity['length']+550);text=passage['text'];context=' '.join((text[left:local]+' <PHENOTYPE> '+text[local:local+entity['length']]+' </PHENOTYPE> '+text[local+entity['length']:right]).split());values=split_value(entity.get('identifier'),entity.get('text',''));names=[]
 for v in values:
  term=ontology.terms.get(v);names.append(f'{v} {term.name if term else ""}')
 allowed=', '.join(str(p['patient_id']) for p in doc['patient'])
 return f"""Identify the experiencer of the single highlighted phenotype occurrence, using only the supplied article context.

PATIENTS AND KNOWN ALIASES:
{chr(10).join(patients)}

PHENOTYPE: {'; '.join(names)}
SECTION: {passage.get('section_type')}
CONTEXT: {context}

Rules:
- Return one or more exact patient IDs only when this occurrence explicitly describes those patients.
- Return BACKGROUND for general knowledge, literature, disease descriptions, or non-case discussion.
- Return NONE when it explicitly concerns an unlisted relative, control, or other person.
- Return UNCERTAIN when the experiencer cannot be resolved from this context.
- Do not infer from diagnosis, gene, or medical knowledge.

Allowed patient IDs: {allowed}
Return exactly either a comma-separated list of allowed patient IDs, BACKGROUND, NONE, or UNCERTAIN."""
def parse(text,patient_ids):
 upper=text.strip();special=next((x for x in ['BACKGROUND','UNCERTAIN','NONE'] if re.search(rf'\b{x}\b',upper,re.I)),None)
 found=[]
 for pid in sorted(patient_ids,key=len,reverse=True):
  if re.search(rf'(?<![A-Za-z0-9_]){re.escape(pid)}(?![A-Za-z0-9_])',upper):found.append(pid)
 return {'kind':'PATIENT','patients':found} if found else {'kind':special or 'UNCERTAIN','patients':[]}
def main():
 p=argparse.ArgumentParser();p.add_argument('--fold',type=int,default=0);p.add_argument('--model',default=str(ROOT/'data/models/Qwen3-8B-hf'));p.add_argument('--output-dir',type=Path,default=ROOT/'data/processed/e17/fold0');a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True);raw=ROOT/'data/raw/PatientPheX-V1-A';docs=load_jsonl(str(raw/'PatientPheX-train.jsonl'));by={str(d['pmc_id']):d for d in docs};split=json.loads((ROOT/'data/processed/e0/splits.json').read_text())['cv_folds'][a.fold];train=[by[x] for x in split['train']];gold=[by[x] for x in split['validation']];entity_rows={str(d['pmc_id']):d['entities'] for d in load_jsonl(str(ROOT/'data/processed/e14/gated_oof'/f'fold{a.fold}.jsonl'))};validation=[]
 for doc in gold:item=deepcopy(doc);item['entities']=entity_rows[str(doc['pmc_id'])];validation.append(item)
 ontology=HPOntology(str(raw/'hp.obo'));cache_path=a.output_dir/'owner_cache.jsonl';cache={}
 if cache_path.exists():
  for line in cache_path.read_text().splitlines():row=json.loads(line);cache[tuple(row['key'])]=row
 tasks=[]
 for doc in validation:
  for e in doc['entities']:
   key=(str(doc['pmc_id']),int(e['offset']),int(e['length']),str(e['identifier']))
   if key not in cache:tasks.append((doc,e,key))
 print('occurrences',sum(len(d['entities']) for d in validation),'pending',len(tasks),flush=True)
 if tasks:
  tokenizer=AutoTokenizer.from_pretrained(a.model,padding_side='left');model=AutoModelForCausalLM.from_pretrained(a.model,dtype=torch.bfloat16,device_map='cuda',attn_implementation='sdpa').eval()
  for start in range(0,len(tasks),8):
   batch_tasks=tasks[start:start+8];prompts=[]
   for doc,e,key in batch_tasks:
    messages=[{'role':'system','content':'You are a precise biomedical evidence attribution classifier.'},{'role':'user','content':prompt(doc,e,ontology)}];prompts.append(tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False))
   encoded=tokenizer(prompts,return_tensors='pt',padding=True,truncation=True,max_length=3072).to(model.device)
   with torch.inference_mode():generated=model.generate(**encoded,max_new_tokens=24,do_sample=False,pad_token_id=tokenizer.eos_token_id)
   rows=[];input_length=encoded['input_ids'].shape[1]
   for output,(doc,e,key) in zip(generated,batch_tasks):
    raw_output=tokenizer.decode(output[input_length:],skip_special_tokens=True).strip();parsed=parse(raw_output,[str(x['patient_id']) for x in doc['patient']]);row={'key':list(key),'raw':raw_output,**parsed};rows.append(row);cache[key]=row
   with cache_path.open('a') as handle:
    for row in rows:handle.write(json.dumps(row,ensure_ascii=False)+'\n')
   print('completed',min(start+len(batch_tasks),len(tasks)),'/',len(tasks),flush=True)
 xt,yt,_=build_examples(train);xv,_,keys=build_examples(validation);prob=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000,class_weight='balanced',C=1,random_state=2026)).fit(xt,yt).predict_proba(xv)[:,1];pair_cache=load_cache(ROOT/f'data/processed/e15/fold{a.fold}/llm_cache.jsonl');base={}
 for key,z in zip(map(tuple,keys),prob):base[key]=(pair_cache[key]['label']=='YES') if key in pair_cache else z>=.65
 owner=defaultdict(set);decisive_concepts=set();kinds=defaultdict(int)
 for doc in validation:
  pmc=str(doc['pmc_id'])
  for e in doc['entities']:
   row=cache[(pmc,int(e['offset']),int(e['length']),str(e['identifier']))];kinds[row['kind']]+=1
   if row['kind']=='PATIENT':
    for value in split_value(e.get('identifier'),e.get('text','')):
     decisive_concepts.add((pmc,value))
     for pid in row['patients']:owner[(pmc,pid)].add(value)
 strategies={}
 strategies['owner_only']={(pmc,pid,value) for (pmc,pid),values in owner.items() for value in values}
 strategies['union']=set(k for k,v in base.items() if v)|strategies['owner_only']
 strategies['add_when_base_empty']=set(k for k,v in base.items() if v)
 base_concepts={(k[0],k[2]) for k,v in base.items() if v}
 strategies['add_when_base_empty'].update(k for k in strategies['owner_only'] if (k[0],k[2]) not in base_concepts)
 strategies['replace_when_decisive']={k for k,v in base.items() if v and (k[0],k[2]) not in decisive_concepts}|strategies['owner_only'];entities={str(d['pmc_id']):d['entities'] for d in validation};reports={}
 for name,chosen in strategies.items():
  rows={k:{'label':'YES' if k in chosen else 'NO'} for k in map(tuple,keys)};pred=relation_prediction(validation,entities,keys,prob,.65,rows,'negative');reports[name]=compute_all_metrics(gold,pred);print(name,reports[name]['overall_score'],reports[name]['subtask2']['f1_micro'],reports[name]['subtask2']['f1_macro'])
 report={'scope':f'CV fold {a.fold}; locked validation untouched','output_kinds':dict(kinds),'metrics':reports};(a.output_dir/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
if __name__=='__main__':main()
