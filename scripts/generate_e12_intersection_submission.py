#!/usr/bin/env python3
"""Create A submission using the frozen E7 ∩ two-seed E12 rule."""
from copy import deepcopy
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.data_loader import load_jsonl,save_jsonl
from src.submission import validate_submission,raise_for_errors
raw=ROOT/'data/raw/PatientPheX-V1-A';source=load_jsonl(str(raw/'PatientPheX-A.jsonl'));e7=load_jsonl(str(ROOT/'submissions/patientphex_a_frozen_e7.jsonl'));p1={tuple(x['key']):x['probability'] for x in map(json.loads,open(ROOT/'data/processed/final/e12_seed2026/a_probabilities.jsonl'))};p2={tuple(x['key']):x['probability'] for x in map(json.loads,open(ROOT/'data/processed/final/e12_seed2027/a_probabilities.jsonl'))};submission=[];before=after=0
for doc in e7:
 item=deepcopy(doc);pmc=str(item['pmc_id'])
 for association in item['association']:
  patient=str(association['patient_id']);old=list(association['phenotype']);before+=len(old);kept=[]
  for value in old:
   key=(pmc,patient,value)
   if key not in p1 or key not in p2:raise KeyError(f'missing E12 probability for {key}')
   if (p1[key]+p2[key])/2>=.20:kept.append(value)
  association['phenotype']=kept;after+=len(kept)
 submission.append(item)
issues=validate_submission(source,submission);raise_for_errors(issues);path=ROOT/'submissions/patientphex_a_e12_intersection.jsonl';save_jsonl(submission,str(path));summary={'rule':'E7 positive AND mean(E12 seeds 2026,2027) >= 0.20','documents':len(submission),'associations_before':before,'associations_after':after,'removed':before-after,'validation_warnings':[x.to_dict() for x in issues],'output':str(path)};(ROOT/'data/processed/final/e12_submission_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');print(json.dumps(summary,ensure_ascii=False,indent=2))
