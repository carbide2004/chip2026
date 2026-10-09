#!/usr/bin/env python3
"""Four-fold OOF Marker/SapBERT threshold stability search; no locked data."""
from collections import defaultdict
import argparse,itertools,json,statistics,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from transformers import AutoModel,AutoTokenizer
from src.baseline_v1 import LexicalPhenotypeBaseline,build_prediction
from src.data_loader import HPOntology,load_jsonl
from src.hpo_retrieval import LexicalHPORetriever
from src.metrics import compute_all_metrics
MARKERS=[.45,.475,.5,.525,.55,.575,.6,.625,.65,.675,.7,.725,.75]
SAPS=[.84,.85,.86,.87,.88,.89,.9,.91,.92,.93,.94]
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--candidate-root',type=Path,default=None);parser.add_argument('--output-file',type=Path,default=ROOT/'data/processed/e15/entity_tuning_report.json');args=parser.parse_args()
 raw=ROOT/'data/raw/PatientPheX-V1-A';docs=load_jsonl(str(raw/'PatientPheX-train.jsonl'));by={str(d['pmc_id']):d for d in docs};splits=json.loads((ROOT/'data/processed/e0/splits.json').read_text());onto=HPOntology(str(raw/'hp.obo'));lex=LexicalPhenotypeBaseline(onto);ret=LexicalHPORetriever(onto);fold_candidates={};surfaces=set()
 def surface(doc,row):
  for p in doc['full_text']:
   a=row['offset']-p['offset'];b=row['end']-p['offset']
   if 0<=a<b<=len(p['text']):return p['text'][a:b]
  return ''
 for fold in range(4):
  path=(args.candidate_root/f'fold{fold}/candidate_probabilities.jsonl') if args.candidate_root else ROOT/f'data/processed/e14/{"marker_fold0_joint" if fold==0 else f"marker_fold{fold}"}/candidate_probabilities.jsonl';fold_candidates[fold]=defaultdict(list)
  for row in map(json.loads,open(path)):
   row['surface']=surface(by[str(row['pmc_id'])],row);fold_candidates[fold][str(row['pmc_id'])].append(row);surfaces.add(ret.normalize(row['surface']))
 name='cambridgeltl/SapBERT-from-PubMedBERT-fulltext';tok=AutoTokenizer.from_pretrained(name);model=AutoModel.from_pretrained(name).to('cuda').eval()
 def enc(texts,batch=256):
  output=[]
  with torch.inference_mode():
   for i in range(0,len(texts),batch):
    x=tok(texts[i:i+batch],padding=True,truncation=True,max_length=32,return_tensors='pt').to('cuda');output.append(torch.nn.functional.normalize(model(**x).last_hidden_state[:,0].float(),p=2,dim=1).cpu())
  return torch.cat(output)
 phrases=ret.phrases;pv=enc(phrases).to('cuda');surface_list=sorted(surfaces);q=enc(surface_list);scores,indices=(q.to('cuda')@pv.T).max(1);mapping={s:(sorted(ret.phrase_to_ids[phrases[i]])[0],float(z)) for s,i,z in zip(surface_list,indices.cpu().tolist(),scores.cpu().tolist())};results=[]
 for mt,st in itertools.product(MARKERS,SAPS):
  fold_rows=[]
  for fold in range(4):
   gold=[by[x] for x in splits['cv_folds'][fold]['validation']];predictions=[]
   for doc in gold:
    entities=lex.extract_document(doc,overlap_policy='longest',detect_negation=False);occupied={(e['offset'],e['length']) for e in entities}
    for row in fold_candidates[fold][str(doc['pmc_id'])]:
     identifier,score=mapping[ret.normalize(row['surface'])];key=(row['offset'],row['end']-row['offset'])
     if row['probability']<mt or score<st or key in occupied:continue
     entities.append({'identifier':identifier,'type':'Phenotype','offset':key[0],'length':key[1],'text':row['surface'],'note':None});occupied.add(key)
    entities.sort(key=lambda e:(e['offset'],e['length'],e['identifier']));predictions.append(build_prediction(doc,entities,association_method='same_passage',association_window=0))
   m=compute_all_metrics(gold,predictions);fold_rows.append({'mention':m['subtask1']['f1_men'],'document':m['subtask1']['f1_doc'],'entity_mean':(m['subtask1']['f1_men']+m['subtask1']['f1_doc'])/2})
  means=[x['entity_mean'] for x in fold_rows];results.append({'marker_threshold':mt,'sapbert_threshold':st,'mean_entity':statistics.mean(means),'min_entity':min(means),'folds':fold_rows,'mean_mention':statistics.mean(x['mention'] for x in fold_rows),'mean_document':statistics.mean(x['document'] for x in fold_rows)})
 results.sort(key=lambda x:(x['mean_entity'],x['min_entity']),reverse=True);baseline=next(x for x in results if x['marker_threshold']==.6 and x['sapbert_threshold']==.9);report={'scope':'four-fold development OOF only; locked validation untouched','baseline':baseline,'top':results[:50]};args.output_file.parent.mkdir(parents=True,exist_ok=True);args.output_file.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print('baseline',baseline)
 for x in results[:20]:print(x)
if __name__=='__main__':main()
