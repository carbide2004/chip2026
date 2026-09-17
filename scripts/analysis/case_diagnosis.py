"""Detailed error diagnosis on train set multi-patient documents comparing gold vs V3 predictions.
Finds exact false positives, false negatives, and hallucinated associations.
"""

import json
import sys
from pathlib import Path

repo_root = Path(r"E:\Workplace_Tanhaowen\chip2026-patientphex")
sys.path.insert(0, str(repo_root))

from src.data_loader import load_jsonl
from src.pipeline_gpu import GPUEnhancedPipeline

docs = load_jsonl(repo_root / "data" / "raw" / "PatientPheX-V1-A" / "PatientPheX-train.jsonl")
multi_docs = [d for d in docs if len(d.get("patient", [])) > 1]

print(f"Analyzing {len(multi_docs)} multi-patient documents from train set...\n")

case_studies = []

for d in multi_docs[:5]:
    pmc = d["pmc_id"]
    patients = d.get("patient", [])
    gold_assoc = {a["patient_id"]: set(a.get("phenotype", [])) for a in d.get("association", [])}
    
    # Check patient mentions
    p_info = []
    for p in patients:
        m_texts = [m["text"] for m in p.get("mention", [])]
        p_info.append(f"{p['patient_id']} ({', '.join(m_texts[:2])})")
    
    # Check how many phenotypes are shared vs exclusive in this document
    all_gold_hpos = set().union(*gold_assoc.values()) if gold_assoc else set()
    shared = [h for h in all_gold_hpos if sum(1 for p, s in gold_assoc.items() if h in s) > 1]
    exclusive = [h for h in all_gold_hpos if sum(1 for p, s in gold_assoc.items() if h in s) == 1]
    
    print(f"=== Doc PMC{pmc} ({len(patients)} patients: {'; '.join(p_info)}) ===")
    for p, hpos in gold_assoc.items():
        print(f"  Patient '{p}': {len(hpos)} gold phenotypes")
    print(f"  Shared phenotypes: {len(shared)}, Exclusive phenotypes: {len(exclusive)}")
    
    # Sample a clinical paragraph from this doc
    case_secs = [s.get("text", "") for s in d.get("full_text", []) if any(k in (s.get("section_type") or "").upper() for k in ["CASE", "CLINICAL", "REPORT", "PRESENT"])]
    if case_secs:
        snippet = case_secs[0][:300].replace("\n", " ")
        print(f"  Clinical Snippet: \"{snippet}...\"\n")
