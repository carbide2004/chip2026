#!/usr/bin/env python3
"""Build repeated four-fold splits over the fixed 64-document development pool."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.data_loader import load_jsonl
from src.e0 import create_balanced_kfold_split,document_features

def main():
 p=argparse.ArgumentParser();p.add_argument('--base-splits',type=Path,default=ROOT/'data/processed/e0/splits.json');p.add_argument('--data-file',type=Path,default=ROOT/'data/raw/PatientPheX-V1-A/PatientPheX-train.jsonl');p.add_argument('--output-file',type=Path,default=ROOT/'data/processed/e26/repeated_splits.json');p.add_argument('--seeds',type=int,nargs='+',default=[3407,7829,104729,99991]);a=p.parse_args();base=json.loads(a.base_splits.read_text());docs=load_jsonl(str(a.data_file));by={str(x['pmc_id']):x for x in docs};development=base['development_pool'];locked=base['locked_validation'];dev_docs=[by[x] for x in development];linked=base.get('near_duplicate_groups',[]);repeats=[{'repeat':0,'seed':base['seed'],'source':'original_e0','strategy':base['strategy'],'cv_folds':base['cv_folds'],'group_features':base['group_features'][1:]}]
 for index,seed in enumerate(a.seeds,1):
  split=create_balanced_kfold_split(dev_docs,k=4,seed=seed,linked_groups=linked);repeats.append({'repeat':index,'seed':seed,'source':'fixed_development_rebalance','strategy':split['strategy'],'cv_folds':split['cv_folds'],'group_features':split['group_features']})
 payload={'scope':'fixed 64-document development pool; 16 locked documents excluded','development_pool':development,'locked_validation':locked,'near_duplicate_groups':linked,'repeats':repeats,'acceptance_gate':{'minimum_mean_overall_gain':.008,'minimum_positive_folds':16,'total_folds':20,'paired_bootstrap_ci_must_exclude_zero':True}}
 a.output_file.parent.mkdir(parents=True,exist_ok=True);a.output_file.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'output':str(a.output_file),'repeats':len(repeats),'fold_sizes':[[len(f['validation']) for f in r['cv_folds']] for r in repeats],'locked':len(locked),'development':len(development)},indent=2))
if __name__=='__main__':main()
