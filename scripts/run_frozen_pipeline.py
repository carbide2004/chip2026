#!/usr/bin/env python3
"""One-command reproducible E15/E7 inference for an unlabeled PatientPheX set."""
from pathlib import Path
import argparse,hashlib,json,shutil,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1]
def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
def run(parts):
 print('+',' '.join(map(str,parts)),flush=True);subprocess.run([str(x) for x in parts],check=True,cwd=ROOT)
def main():
 p=argparse.ArgumentParser();p.add_argument('--input-file',type=Path,required=True);p.add_argument('--train-file',type=Path,default=ROOT/'data/raw/PatientPheX-V1-A/PatientPheX-train.jsonl');p.add_argument('--ontology-file',type=Path,default=ROOT/'data/raw/PatientPheX-V1-A/hp.obo');p.add_argument('--work-dir',type=Path,required=True);p.add_argument('--marker-model-dir',type=Path,default=ROOT/'data/processed/final/marker_e15_model');p.add_argument('--mode',choices=['e15','e7','both'],default='both');p.add_argument('--e15-seed-cache',type=Path);p.add_argument('--e7-seed-cache',type=Path);a=p.parse_args();a.work_dir.mkdir(parents=True,exist_ok=True);started=time.time();py=sys.executable;outputs=[]
 for mode in (['e15','e7'] if a.mode=='both' else [a.mode]):
  entities=a.work_dir/f'{mode}_entities.jsonl';cache=a.work_dir/f'{mode}_qwen_cache.jsonl';submission=a.work_dir/f'patientphex_{mode}.jsonl';summary=a.work_dir/f'{mode}_summary.json';seed=a.e15_seed_cache if mode=='e15' else a.e7_seed_cache
  if seed and seed.exists() and not cache.exists():shutil.copy2(seed,cache)
  if mode=='e15':
   run([py,ROOT/'scripts/train_e14_final.py','--train-file',a.train_file,'--input-file',a.input_file,'--ontology-file',a.ontology_file,'--output-file',entities,'--report-file',a.work_dir/'e15_entity_report.json','--marker-model-dir',a.marker_model_dir,'--reuse-marker-model'])
  else:
   spans=a.work_dir/'e7_spans.jsonl';run([py,ROOT/'scripts/predict_e3_final.py','--input-file',a.input_file,'--output-file',spans]);run([py,ROOT/'scripts/generate_final_a_entities.py','--input-file',a.input_file,'--spans-file',spans,'--ontology-file',a.ontology_file,'--output-file',entities])
  run([py,ROOT/'scripts/generate_final_a_submission.py','--train-file',a.train_file,'--input-file',a.input_file,'--ontology-file',a.ontology_file,'--entities-file',entities,'--cache-file',cache,'--output-file',submission,'--summary-file',summary]);outputs.append({'mode':mode,'submission':str(submission),'sha256':sha(submission),'entities':str(entities),'cache':str(cache),'summary':json.loads(summary.read_text())})
 manifest={'scope':'frozen E15/E7 inference; no test labels used','input':str(a.input_file),'input_sha256':sha(a.input_file),'train_sha256':sha(a.train_file),'ontology_sha256':sha(a.ontology_file),'token_ner_model':{'path':str(ROOT/'data/processed/final/token_ner/model'),'sha256':sha(ROOT/'data/processed/final/token_ner/model/model.safetensors')},'marker_model':{'path':str(a.marker_model_dir),'sha256':sha(a.marker_model_dir/'model.safetensors')},'modes':outputs,'elapsed_seconds':time.time()-started};(a.work_dir/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n');print(json.dumps(manifest,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
