"""Inspect sentences around gold patient mentions to find exact syntax patterns.
"""

import sys
from pathlib import Path
import re

repo_root = Path(r"E:\Workplace_Tanhaowen\chip2026-patientphex")
sys.path.insert(0, str(repo_root))

from src.data_loader import load_jsonl

docs = load_jsonl(repo_root / "data" / "raw" / "PatientPheX-V1-A" / "PatientPheX-train.jsonl")

for d in docs[:10]:
    if len(d.get("patient", [])) <= 1:
        continue
    pmc = d["pmc_id"]
    print(f"=== PMC{pmc} ({len(d.get('patient', []))} patients) ===")
    
    # Extract clinical sentences
    full_text = ""
    for sec in d.get("full_text", []):
        full_text += sec.get("text", "") + "\n"
        
    sents = list(re.finditer(r'[^.!?\n]+[.!?\n]', full_text))
    
    for p in d.get("patient", []):
        pid = p["patient_id"]
        mentions = [m["text"] for m in p.get("mention", [])]
        gold_hpos = next((a["phenotype"] for a in d.get("association", []) if a["patient_id"] == pid), [])
        
        # Find sentences mentioning this patient
        matched_sents = []
        for m in p.get("mention", []):
            m_off = m["offset"]
            s = next((s.group(0).strip() for s in sents if s.start() <= m_off < s.end()), "")
            if s and s not in matched_sents:
                matched_sents.append(s)
                
        print(f"  Patient '{pid}' ({', '.join(mentions[:2])}) -> {len(gold_hpos)} gold phenotypes:")
        for s in matched_sents[:2]:
            print(f"    Excerpt: \"{s[:120]}...\"")
        print()
