#!/usr/bin/env python3
"""Combine lexical entities with frozen high-confidence BIO+SapBERT additions."""
from pathlib import Path
import argparse,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from transformers import AutoModel,AutoTokenizer
from src.baseline_v1 import LexicalPhenotypeBaseline
from src.data_loader import HPOntology,load_jsonl,save_jsonl
from src.hpo_retrieval import LexicalHPORetriever
from src.submission import validate_submission,raise_for_errors

def main():
 p=argparse.ArgumentParser();p.add_argument('--input-file',type=Path,default=ROOT/'data/raw/PatientPheX-V1-A/PatientPheX-A.jsonl');p.add_argument('--spans-file',type=Path,default=ROOT/'data/processed/final/a_span_predictions.jsonl');p.add_argument('--ontology-file',type=Path,default=ROOT/'data/raw/PatientPheX-V1-A/hp.obo');p.add_argument('--output-file',type=Path,default=ROOT/'data/processed/final/a_gated_entities.jsonl');p.add_argument('--sapbert-threshold',type=float,default=.90);a=p.parse_args();docs=load_jsonl(str(a.input_file));spans={str(d['pmc_id']):d['entities'] for d in load_jsonl(str(a.spans_file))};onto=HPOntology(str(a.ontology_file));lex=LexicalPhenotypeBaseline(onto);ret=LexicalHPORetriever(onto);name='cambridgeltl/SapBERT-from-PubMedBERT-fulltext';tok=AutoTokenizer.from_pretrained(name);model=AutoModel.from_pretrained(name).to('cuda').eval()
 def enc(texts,b=256):
  out=[]
  with torch.inference_mode():
   for i in range(0,len(texts),b):
    x=tok(texts[i:i+b],padding=True,truncation=True,max_length=32,return_tensors='pt').to('cuda');out.append(torch.nn.functional.normalize(model(**x).last_hidden_state[:,0].float(),p=2,dim=1).cpu())
  return torch.cat(out)
 phrases=ret.phrases;pv=enc(phrases).to('cuda');surfaces=sorted({ret.normalize(e['text']) for es in spans.values() for e in es});mapping={}
 if surfaces:
  q=enc(surfaces);scores,indices=(q.to('cuda')@pv.T).max(dim=1);mapping={s:(sorted(ret.phrase_to_ids[phrases[i]])[0],float(score)) for s,i,score in zip(surfaces,indices.cpu().tolist(),scores.cpu().tolist())}
 predictions=[];added=0
 for doc in docs:
  entities=lex.extract_document(doc,overlap_policy='longest',detect_negation=False);occupied={(e['offset'],e['length']) for e in entities}
  for e in spans[str(doc['pmc_id'])]:
   if (e['offset'],e['length']) in occupied:continue
   identifier,score=mapping[ret.normalize(e['text'])]
   if score>=a.sapbert_threshold:item=dict(e);item['identifier']=identifier;entities.append(item);occupied.add((e['offset'],e['length']));added+=1
  entities.sort(key=lambda e:(e['offset'],e['length'],e['identifier']));predictions.append({'pmc_id':doc['pmc_id'],'pmid':doc.get('pmid'),'entities':entities,'association':[{'patient_id':p['patient_id'],'phenotype':[]} for p in doc['patient']]})
 issues=validate_submission(docs,predictions);raise_for_errors(issues);a.output_file.parent.mkdir(parents=True,exist_ok=True);save_jsonl(predictions,str(a.output_file));print('documents',len(predictions),'entities',sum(len(d['entities']) for d in predictions),'learned additions',added,'warnings',len(issues),'output',a.output_file)
if __name__=='__main__':main()
