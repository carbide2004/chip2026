#!/usr/bin/env python3
"""Route uncertain patient-concept evidence bags to Qwen3-8B.

The model receives source evidence only and must return YES/NO/UNCERTAIN. Results
are cached after every batch so inference can resume without repeating calls.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import argparse
import json
from pathlib import Path
import re
import sys
import time

REPO_ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO_ROOT))

import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM,AutoTokenizer

from src.baseline_v1 import LexicalPhenotypeBaseline
from src.data_loader import HPOntology,load_jsonl
from src.metrics import compute_all_metrics
from src.relation_features import build_examples,split_value


def predicted_doc(doc,extractor):
 result=deepcopy(doc);result['entities']=extractor.extract_document(doc,overlap_policy='longest',detect_negation=False);return result


def passage_for_offset(doc,offset):
 for passage in doc.get('full_text',[]):
  start=passage.get('offset',0);text=passage.get('text','')
  if start<=offset<start+len(text):return passage
 return None


def evidence_prompt(doc,patient_id,value,ontology,max_occurrences=5):
 patient=next(p for p in doc['patient'] if str(p['patient_id'])==patient_id)
 target_mentions=', '.join(f'"{m["text"]}"@{m["offset"]}' for m in patient.get('mention',[])) or '(none supplied)'
 others=[]
 for p in doc['patient']:
  if str(p['patient_id'])!=patient_id:
   others.append(f'{p["patient_id"]}: '+', '.join(m['text'] for m in p.get('mention',[])[:3]))
 term=ontology.terms.get(value);concept=f'{value} ({term.name})' if term else value
 snippets=[];seen=set()
 for entity in doc.get('entities',[]):
  values=split_value(entity.get('identifier'),entity.get('text',''))
  if value not in values:continue
  passage=passage_for_offset(doc,entity['offset'])
  if not passage:continue
  local=entity['offset']-passage['offset'];left=max(0,local-280);right=min(len(passage['text']),local+entity['length']+280)
  snippet=' '.join(passage['text'][left:right].split())
  key=(passage.get('section_type'),snippet)
  if key not in seen:
   seen.add(key);snippets.append(f'[{passage.get("section_type")}/{passage.get("type")}] ...{snippet}...')
  if len(snippets)>=max_occurrences:break
 return f"""Determine whether the biomedical article explicitly attributes the phenotype concept to the TARGET patient.

Rules:
- YES only when at least one supplied excerpt supports that this target patient has/had the phenotype.
- NO when occurrences concern background knowledge, another relative/patient, a control, or are explicitly negated for the target.
- UNCERTAIN when excerpts are insufficient or conflicting.
- Do not infer a phenotype merely from a diagnosis, gene, or general medical knowledge.
- The supplied target mentions are only partial aliases; resolve obvious pronouns or pedigree labels from context.

TARGET patient: {patient_id}
Supplied target mentions: {target_mentions}
Other listed patients: {'; '.join(others) if others else '(none)'}
Phenotype concept: {concept}

Evidence excerpts:
{chr(10).join(snippets) if snippets else '(no occurrence excerpt found)'}

Return exactly one token: YES, NO, or UNCERTAIN."""


def evidence_prompt_v2(doc,patient_id,value,ontology,max_occurrences=8,max_anchors=5):
 patient=next(p for p in doc['patient'] if str(p['patient_id'])==patient_id);target_mentions=patient.get('mention',[]);all_mentions=[]
 for p in doc['patient']:
  for mention in p.get('mention',[]):all_mentions.append((int(mention['offset']),str(p['patient_id']),mention['text']))
 all_mentions.sort();others=[]
 for p in doc['patient']:
  if str(p['patient_id'])!=patient_id:others.append(f'{p["patient_id"]}: '+', '.join(f'"{m["text"]}"@{m["offset"]}' for m in p.get('mention',[])[:5]))
 term=ontology.terms.get(value);concept=f'{value} ({term.name})' if term else value;occurrences=[];seen=set()
 for entity in doc.get('entities',[]):
  if value not in split_value(entity.get('identifier'),entity.get('text','')):continue
  passage=passage_for_offset(doc,entity['offset'])
  if not passage:continue
  local=entity['offset']-passage['offset'];left=max(0,local-420);right=min(len(passage['text']),local+entity['length']+420);snippet=' '.join(passage['text'][left:right].split());nearest=sorted(all_mentions,key=lambda x:abs(x[0]-entity['offset']))[:3];proximity=', '.join(f'{pid}:"{text}" distance={offset-entity["offset"]:+d}' for offset,pid,text in nearest) or 'none';key=(passage.get('section_type'),snippet)
  if key not in seen:seen.add(key);occurrences.append(f'[{passage.get("section_type")}/{passage.get("type")}; nearest patient anchors: {proximity}] ...{snippet}...')
  if len(occurrences)>=max_occurrences:break
 anchors=[];seen=set()
 for mention in target_mentions[:max_anchors]:
  passage=passage_for_offset(doc,mention['offset'])
  if not passage:continue
  local=mention['offset']-passage['offset'];left=max(0,local-360);right=min(len(passage['text']),local+mention.get('length',len(mention['text']))+360);snippet=' '.join(passage['text'][left:right].split());key=(passage.get('section_type'),snippet)
  if key not in seen:seen.add(key);anchors.append(f'[{passage.get("section_type")}/{passage.get("type")}; anchor="{mention["text"]}"@{mention["offset"]}] ...{snippet}...')
 return f"""Classify whether the article explicitly attributes the HPO phenotype to the TARGET patient.

Decision rules:
- YES: textual evidence says the TARGET patient has or had this phenotype.
- NO: evidence is about another patient/relative, background knowledge, a control, or explicitly denies it for TARGET.
- UNCERTAIN: evidence does not identify the experiencer or is genuinely conflicting.
- Never infer phenotype from diagnosis, gene, treatment, test purpose, or general medical knowledge.
- Resolve pedigree labels, patient numbering, kinship words, and pronouns using the supplied contexts.
- A nearby patient anchor is useful but not conclusive; determine the grammatical/clinical experiencer.

TARGET: {patient_id}
TARGET aliases: {', '.join(f'"{m["text"]}"@{m["offset"]}' for m in target_mentions) or '(none supplied)'}
OTHER patients: {'; '.join(others) if others else '(none)'}
PHENOTYPE: {concept}

PHENOTYPE OCCURRENCE CONTEXTS:
{chr(10).join(occurrences) if occurrences else '(none)'}

TARGET ANCHOR CONTEXTS:
{chr(10).join(anchors) if anchors else '(none)'}

Return exactly one token: YES, NO, or UNCERTAIN."""


def parse_label(text):
 matches=re.findall(r'\b(YES|NO|UNCERTAIN)\b',text.upper())
 return matches[-1] if matches else 'UNCERTAIN'


def load_cache(path):
 output={}
 if path.exists():
  for line in path.read_text(encoding='utf-8').splitlines():
   if line.strip():
    row=json.loads(line);output[tuple(row['key'])]=row
 return output


def append_cache(path,rows):
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('a',encoding='utf-8') as handle:
  for row in rows:handle.write(json.dumps(row,ensure_ascii=False)+'\n')


def relation_prediction(docs,entities_by_doc,keys,probabilities,threshold,llm_rows=None,uncertain_policy='base'):
 llm_rows=llm_rows or {};selected=defaultdict(list)
 for key,probability in zip(keys,probabilities):
  decision=probability>=threshold
  row=llm_rows.get(tuple(key))
  if row:
   if row['label']=='YES':decision=True
   elif row['label']=='NO':decision=False
   elif uncertain_policy=='negative':decision=False
  if decision:selected[(key[0],key[1])].append(key[2])
 output=[]
 for doc in docs:
  pmc_id=str(doc['pmc_id'])
  output.append({'pmc_id':doc['pmc_id'],'pmid':doc.get('pmid'),'entities':entities_by_doc[pmc_id],'association':[{'patient_id':p['patient_id'],'phenotype':selected[(pmc_id,str(p['patient_id']))]} for p in doc['patient']]})
 return output


def main():
 p=argparse.ArgumentParser();p.add_argument('--model',default='Qwen/Qwen3-8B');p.add_argument('--fold',type=int,default=0);p.add_argument('--locked-validation',action='store_true');p.add_argument('--prompt-version',choices=['v1','v2'],default='v1');p.add_argument('--low',type=float,default=.35);p.add_argument('--high',type=float,default=.85);p.add_argument('--base-threshold',type=float,default=.65);p.add_argument('--batch-size',type=int,default=8)
 p.add_argument('--data-dir',type=Path,default=REPO_ROOT/'data/raw/PatientPheX-V1-A');p.add_argument('--e0-dir',type=Path,default=REPO_ROOT/'data/processed/e0');p.add_argument('--validation-entities',type=Path);p.add_argument('--output-dir',type=Path,default=REPO_ROOT/'data/processed/e6/fold0');args=p.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True);started=time.time()
 docs=load_jsonl(str(args.data_dir/'PatientPheX-train.jsonl'));by_id={str(d['pmc_id']):d for d in docs};splits=json.loads((args.e0_dir/'splits.json').read_text())
 if args.locked_validation:
  train_ids=splits['development_pool'];validation_ids=splits['locked_validation'];evaluation_scope='locked validation'
 else:
  split=splits['cv_folds'][args.fold];train_ids=split['train'];validation_ids=split['validation'];evaluation_scope=f'CV fold {args.fold}'
 ontology=HPOntology(str(args.data_dir/'hp.obo'));extractor=LexicalPhenotypeBaseline(ontology)
 train=[by_id[x] for x in train_ids];validation_gold=[by_id[x] for x in validation_ids]
 if args.validation_entities:
  supplied={str(d['pmc_id']):d['entities'] for d in load_jsonl(str(args.validation_entities))};validation_pred=[]
  for doc in validation_gold:
   item=deepcopy(doc);item['entities']=supplied[str(doc['pmc_id'])];validation_pred.append(item)
  entity_source=str(args.validation_entities)
 else:
  validation_pred=[predicted_doc(d,extractor) for d in validation_gold];entity_source='lexical'
 entities_by_doc={str(d['pmc_id']):d['entities'] for d in validation_pred}
 x_train,y_train,_=build_examples(train);x_val,y_val,keys=build_examples(validation_pred);classifier=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000,class_weight='balanced',C=1.0,random_state=2026)).fit(x_train,y_train);probabilities=classifier.predict_proba(x_val)[:,1]
 query_indices=[i for i,z in enumerate(probabilities) if args.low<=z<=args.high];cache_path=args.output_dir/'llm_cache.jsonl';cache=load_cache(cache_path);pending=[i for i in query_indices if tuple(keys[i]) not in cache]
 print(f'candidate pairs={len(keys)}, routed={len(query_indices)}, cached={len(query_indices)-len(pending)}, pending={len(pending)}',flush=True)

 if pending:
  tokenizer=AutoTokenizer.from_pretrained(args.model,padding_side='left');model=AutoModelForCausalLM.from_pretrained(args.model,dtype=torch.bfloat16,device_map='cuda',attn_implementation='sdpa');model.eval();revision=getattr(model.config,'_commit_hash',None);parameter_count=sum(x.numel() for x in model.parameters())
  for start in range(0,len(pending),args.batch_size):
   indices=pending[start:start+args.batch_size];prompts=[]
   for index in indices:
    pmc_id,patient_id,value=keys[index];messages=[{'role':'system','content':'You are a precise biomedical relation classifier. Follow the requested output format.'},{'role':'user','content':(evidence_prompt_v2 if args.prompt_version=='v2' else evidence_prompt)(next(d for d in validation_pred if str(d['pmc_id'])==pmc_id),patient_id,value,ontology)}]
    prompts.append(tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False))
   batch=tokenizer(prompts,return_tensors='pt',padding=True,truncation=True,max_length=3072 if args.prompt_version=='v2' else 2048).to(model.device)
   with torch.inference_mode():generated=model.generate(**batch,max_new_tokens=4,do_sample=False,pad_token_id=tokenizer.eos_token_id)
   new=[]
   for row,index,input_length in zip(generated,indices,[batch['input_ids'].shape[1]]*len(indices)):
    raw=tokenizer.decode(row[input_length:],skip_special_tokens=True).strip();key=keys[index];item={'key':list(key),'probability':float(probabilities[index]),'gold_candidate_label':int(y_val[index]),'label':parse_label(raw),'raw':raw};new.append(item);cache[tuple(key)]=item
   append_cache(cache_path,new);print(f'completed {min(start+len(indices),len(pending))}/{len(pending)}',flush=True)
 else:
  revision=None;parameter_count=None

 base=relation_prediction(validation_pred,entities_by_doc,keys,probabilities,args.base_threshold)
 routed_base=relation_prediction(validation_pred,entities_by_doc,keys,probabilities,args.base_threshold,cache,'base')
 routed_negative=relation_prediction(validation_pred,entities_by_doc,keys,probabilities,args.base_threshold,cache,'negative')
 queried=[cache[tuple(keys[i])] for i in query_indices];label_counts=defaultdict(int)
 for row in queried:label_counts[row['label']]+=1
 report={'scope':f'{evaluation_scope}; entity source={entity_source}','model':{'name':args.model,'revision':revision or 'from cache','parameter_count':parameter_count},'routing':{'prompt_version':args.prompt_version,'low':args.low,'high':args.high,'base_threshold':args.base_threshold,'queried':len(query_indices),'label_counts':dict(label_counts)},'metrics':{'base_logistic':compute_all_metrics(validation_gold,base),'llm_uncertain_keep_base':compute_all_metrics(validation_gold,routed_base),'llm_uncertain_as_negative':compute_all_metrics(validation_gold,routed_negative)},'queried_subset':{'count':len(queried),'gold_positive':sum(x['gold_candidate_label'] for x in queried),'accuracy_excluding_uncertain':sum((x['label']=='YES')==bool(x['gold_candidate_label']) for x in queried if x['label']!='UNCERTAIN')/max(1,sum(x['label']!='UNCERTAIN' for x in queried))},'elapsed_seconds':time.time()-started}
 (args.output_dir/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
