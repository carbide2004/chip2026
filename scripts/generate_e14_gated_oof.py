#!/usr/bin/env python3
"""Generate four-fold OOF entities with frozen marker=.60 and SapBERT=.90 gates."""
from collections import defaultdict
import argparse,json,statistics,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from transformers import AutoModel,AutoTokenizer
from src.baseline_v1 import LexicalPhenotypeBaseline,build_prediction
from src.data_loader import HPOntology,load_jsonl,save_jsonl
from src.hpo_retrieval import LexicalHPORetriever
from src.metrics import compute_all_metrics
parser=argparse.ArgumentParser();parser.add_argument('--marker-threshold',type=float,default=.60);parser.add_argument('--sapbert-threshold',type=float,default=.90);parser.add_argument('--candidate-root',type=Path,default=None);parser.add_argument('--output-dir',type=Path,default=ROOT/'data/processed/e14/gated_oof');args=parser.parse_args()
raw=ROOT/'data/raw/PatientPheX-V1-A';docs=load_jsonl(str(raw/'PatientPheX-train.jsonl'));by={str(d['pmc_id']):d for d in docs};splits=json.loads((ROOT/'data/processed/e0/splits.json').read_text());onto=HPOntology(str(raw/'hp.obo'));lex=LexicalPhenotypeBaseline(onto);ret=LexicalHPORetriever(onto);out=args.output_dir;out.mkdir(parents=True,exist_ok=True);fold_candidates={};surfaces=set()
def surface(doc,row):
 for p in doc['full_text']:
  a=row['offset']-p['offset'];b=row['end']-p['offset']
  if 0<=a<b<=len(p['text']):return p['text'][a:b]
 return ''
for fold in range(4):
 path=(args.candidate_root/f'fold{fold}/candidate_probabilities.jsonl') if args.candidate_root else ROOT/f'data/processed/e14/{"marker_fold0_joint" if fold==0 else f"marker_fold{fold}"}/candidate_probabilities.jsonl';rows=[json.loads(x) for x in open(path)];fold_candidates[fold]=defaultdict(list)
 for row in rows:
  text=surface(by[str(row['pmc_id'])],row);row['surface']=text;fold_candidates[fold][str(row['pmc_id'])].append(row);surfaces.add(ret.normalize(text))
name='cambridgeltl/SapBERT-from-PubMedBERT-fulltext';tok=AutoTokenizer.from_pretrained(name);model=AutoModel.from_pretrained(name).to('cuda').eval()
def enc(texts,b=256):
 output=[]
 with torch.inference_mode():
  for i in range(0,len(texts),b):
   x=tok(texts[i:i+b],padding=True,truncation=True,max_length=32,return_tensors='pt').to('cuda');v=model(**x).last_hidden_state[:,0].float();output.append(torch.nn.functional.normalize(v,p=2,dim=1).cpu())
 return torch.cat(output)
phrases=ret.phrases;pv=enc(phrases).to('cuda');surface_list=sorted(surfaces);q=enc(surface_list);scores,indices=(q.to('cuda')@pv.T).max(1);mapping={s:(sorted(ret.phrase_to_ids[phrases[i]])[0],float(z)) for s,i,z in zip(surface_list,indices.cpu().tolist(),scores.cpu().tolist())};reports=[]
for fold in range(4):
 gold=[by[x] for x in splits['cv_folds'][fold]['validation']];predictions=[];added=0
 for doc in gold:
  entities=lex.extract_document(doc,overlap_policy='longest',detect_negation=False);occupied={(e['offset'],e['length']) for e in entities}
  for row in fold_candidates[fold][str(doc['pmc_id'])]:
   if row['probability']<args.marker_threshold:continue
   identifier,score=mapping[ret.normalize(row['surface'])]
   if score<args.sapbert_threshold:continue
   key=(row['offset'],row['end']-row['offset'])
   if key in occupied:continue
   entities.append({'identifier':identifier,'type':'Phenotype','offset':key[0],'length':key[1],'text':row['surface'],'note':None});occupied.add(key);added+=1
  entities.sort(key=lambda e:(e['offset'],e['length'],e['identifier']));predictions.append(build_prediction(doc,entities,association_method='same_passage',association_window=0))
 save_jsonl(predictions,str(out/f'fold{fold}.jsonl'));m=compute_all_metrics(gold,predictions);reports.append({'fold':fold,'added':added,'metrics':m});print(fold,added,m['overall_score'],m['subtask1']['f1_men'],m['subtask1']['f1_doc'])
summary={'scope':f'four-fold OOF; marker threshold {args.marker_threshold} and SapBERT threshold {args.sapbert_threshold}','folds':reports,'mean':{'overall':statistics.mean(x['metrics']['overall_score'] for x in reports),'mention':statistics.mean(x['metrics']['subtask1']['f1_men'] for x in reports),'document':statistics.mean(x['metrics']['subtask1']['f1_doc'] for x in reports),'micro':statistics.mean(x['metrics']['subtask2']['f1_micro'] for x in reports),'macro':statistics.mean(x['metrics']['subtask2']['f1_macro'] for x in reports)}};(out/'report.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');print(json.dumps(summary['mean'],indent=2))
