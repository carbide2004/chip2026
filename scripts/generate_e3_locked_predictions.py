#!/usr/bin/env python3
"""Generate frozen gated entity predictions for the one-time locked validation."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from transformers import AutoModel,AutoTokenizer
from src.baseline_v1 import LexicalPhenotypeBaseline
from src.data_loader import HPOntology,load_jsonl,save_jsonl
from src.hpo_retrieval import LexicalHPORetriever
from src.submission import validate_submission,raise_for_errors
raw=ROOT/'data/raw/PatientPheX-V1-A';docs=load_jsonl(str(raw/'PatientPheX-train.jsonl'));by={str(d['pmc_id']):d for d in docs};splits=json.loads((ROOT/'data/processed/e0/splits.json').read_text());source=[by[x] for x in splits['locked_validation']];spans={str(d['pmc_id']):d['entities'] for d in load_jsonl(str(ROOT/'data/processed/e3/token_ner_locked/span_predictions_calibrated.jsonl'))};onto=HPOntology(str(raw/'hp.obo'));lex=LexicalPhenotypeBaseline(onto);ret=LexicalHPORetriever(onto);device='cuda' if torch.cuda.is_available() else 'cpu';name='cambridgeltl/SapBERT-from-PubMedBERT-fulltext';tok=AutoTokenizer.from_pretrained(name);model=AutoModel.from_pretrained(name).to(device).eval()
def enc(texts,b=256):
 out=[]
 with torch.inference_mode():
  for i in range(0,len(texts),b):
   x=tok(texts[i:i+b],padding=True,truncation=True,max_length=32,return_tensors='pt').to(device);v=model(**x).last_hidden_state[:,0].float();out.append(torch.nn.functional.normalize(v,p=2,dim=1).cpu())
 return torch.cat(out)
phrases=ret.phrases;pv=enc(phrases).to(device);surfaces=sorted({ret.normalize(e['text']) for es in spans.values() for e in es});q=enc(surfaces);scores,indices=(q.to(device)@pv.T).max(dim=1);mapping={s:(sorted(ret.phrase_to_ids[phrases[i]])[0],float(score)) for s,i,score in zip(surfaces,indices.cpu().tolist(),scores.cpu().tolist())};predictions=[];added=0
for doc in source:
 entities=lex.extract_document(doc,overlap_policy='longest',detect_negation=False);occupied={(e['offset'],e['length']) for e in entities}
 for e in spans[str(doc['pmc_id'])]:
  if (e['offset'],e['length']) in occupied:continue
  identifier,score=mapping[ret.normalize(e['text'])]
  if score>=.90:item=dict(e);item['identifier']=identifier;entities.append(item);added+=1
 entities.sort(key=lambda e:(e['offset'],e['length'],e['identifier']));predictions.append({'pmc_id':doc['pmc_id'],'pmid':doc.get('pmid'),'entities':entities,'association':[{'patient_id':p['patient_id'],'phenotype':[]} for p in doc['patient']]})
issues=validate_submission(source,predictions);raise_for_errors(issues);out=ROOT/'data/processed/e3/gated_locked.jsonl';save_jsonl(predictions,str(out));print('locked docs',len(predictions),'learned additions',added,'warnings',len(issues),'output',out)
