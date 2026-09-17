"""Inspect exact cases where Qwen3-8B made filtering decisions in V3.
"""

import json
from pathlib import Path

repo_root = Path(r"E:\Workplace_Tanhaowen\chip2026-patientphex")

with open(repo_root / "submissions" / "sapbert_rule_A.jsonl", "r", encoding="utf-8") as f:
    sap = [json.loads(l) for l in f]
with open(repo_root / "submissions" / "qwen3_8b_v3_balanced_A.jsonl", "r", encoding="utf-8") as f:
    v3 = [json.loads(l) for l in f]

for i in range(len(sap)):
    pmc = sap[i]["pmc_id"]
    p_sap = {a["patient_id"]: set(a["phenotype"]) for a in sap[i]["association"]}
    p_v3 = {a["patient_id"]: set(a["phenotype"]) for a in v3[i]["association"]}
    
    diffs = []
    for pid in p_sap:
        removed = p_sap[pid] - p_v3[pid]
        if removed:
            diffs.append((pid, removed))
            
    if diffs:
        print(f"=== Doc {i+1} PMC{pmc} ({len(p_sap)} patients) ===")
        for pid, rem in diffs:
            print(f"  Patient '{pid}': Filtered {len(rem)} phenotypes -> {list(rem)[:5]}")
        print()
