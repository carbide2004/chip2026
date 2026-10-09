#!/usr/bin/env python3
"""Evaluate a leakage-safe fold-trained phenotype surface lexicon."""
from collections import Counter,defaultdict
from copy import deepcopy
import json,re,statistics,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.baseline_v1 import build_prediction
from src.data_loader import load_jsonl,save_jsonl
from src.metrics import compute_all_metrics
raw=ROOT/'data/raw/PatientPheX-V1-A';docs=load_jsonl(str(raw/'PatientPheX-train.jsonl'));by={str(d['pmc_id']):d for d in docs};splits=json.loads((ROOT/'data/processed/e0/splits.json').read_text());out=ROOT/'data/processed/e9';out.mkdir(parents=True,exist_ok=True);reports=[]
def norm(s):return ' '.join(s.casefold().split())
for fold in range(4):
 split=splits['cv_folds'][fold];surface_ids=defaultdict(Counter)
 for pmc in split['train']:
  for e in by[pmc]['entities']:
   if not e.get('negated',False) and e.get('identifier') and e['identifier']!='-1':surface_ids[norm(e['text'])][e['identifier']]+=1
 lexicon={s:c.most_common(1)[0][0] for s,c in surface_ids.items() if len(s)>=3 and len(c)==1};patterns=[(s,identifier,re.compile(re.escape(s),re.I)) for s,identifier in sorted(lexicon.items(),key=lambda x:-len(x[0]))];base_rows={str(d['pmc_id']):d for d in load_jsonl(str(ROOT/'data/processed/e3/gated_oof'/f'fold{fold}.jsonl'))};predictions=[];added=0
 for_doc=[]
 for pmc in split['validation']:
  doc=by[pmc];entities=deepcopy(base_rows[pmc]['entities']);occupied={(e['offset'],e['length']) for e in entities}
  for passage in doc['full_text']:
   text=passage['text']
   for surface,identifier,pattern in patterns:
    for match in pattern.finditer(text):
     start=passage['offset']+match.start();key=(start,match.end()-match.start())
     if key in occupied:continue
     before=text[match.start()-1:match.start()] if match.start() else '';after=text[match.end():match.end()+1]
     if (before and before.isalnum()) or (after and after.isalnum()):continue
     entities.append({'identifier':identifier,'type':'Phenotype','offset':start,'length':key[1],'text':match.group(),'note':None});occupied.add(key);added+=1
  entities.sort(key=lambda e:(e['offset'],e['length'],e['identifier']));predictions.append(build_prediction(doc,entities,association_method='same_passage',association_window=0))
 save_jsonl(predictions,str(out/f'fold{fold}.jsonl'));gold=[by[x] for x in split['validation']];metrics=compute_all_metrics(gold,predictions);reports.append({'fold':fold,'lexicon_size':len(lexicon),'added':added,'metrics':metrics});print(fold,len(lexicon),added,metrics['overall_score'],metrics['subtask1']['f1_men'],metrics['subtask1']['f1_doc'])
summary={'scope':'four-fold OOF; each corpus lexicon built only from that fold training documents','folds':reports,'mean':{'overall':statistics.mean(x['metrics']['overall_score'] for x in reports),'mention':statistics.mean(x['metrics']['subtask1']['f1_men'] for x in reports),'document':statistics.mean(x['metrics']['subtask1']['f1_doc'] for x in reports),'micro':statistics.mean(x['metrics']['subtask2']['f1_micro'] for x in reports),'macro':statistics.mean(x['metrics']['subtask2']['f1_macro'] for x in reports)}};(out/'report.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');print(json.dumps(summary['mean'],indent=2))
