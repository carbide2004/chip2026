import json
from pathlib import Path

repo_root = Path(r"E:\Workplace_Tanhaowen\chip2026-patientphex")

with open(repo_root / "data" / "raw" / "PatientPheX-V1-A" / "PatientPheX-A.jsonl", "r", encoding="utf-8") as f:
    docs = [json.loads(l) for l in f]

# Let's inspect Doc 5 (PMC12078196) and Doc 8 (PMC12711360)
for idx in [4, 7]:
    d = docs[idx]
    pmc = d["pmc_id"]
    print(f"==================================================")
    print(f"Doc {idx+1} PMC{pmc} ({len(d.get('patient', []))} patients)")
    for p in d.get("patient", []):
        m_txts = [m["text"] for m in p.get("mention", [])]
        print(f"  Patient '{p['patient_id']}': {m_txts}")
    
    # Print clinical excerpt
    secs = [s.get("text", "") for s in d.get("full_text", []) if any(k in (s.get("section_type") or "").upper() for k in ["CASE", "CLINICAL", "PRESENT"])]
    if secs:
        print("\nClinical Excerpt:\n" + secs[0][:600].replace("\n", " ") + "\n")
