import json
from pathlib import Path
repo_root = Path(r"E:\Workplace_Tanhaowen\chip2026-patientphex")

docs = []
with open(repo_root / "data" / "raw" / "PatientPheX-V1-A" / "PatientPheX-A.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        docs.append(json.loads(line))

for i, doc in enumerate(docs[:5]):
    patients = doc.get("patient", [])
    print(f"Doc {i+1} PMC{doc['pmc_id']}: {len(patients)} patients")
    for p in patients:
        m_texts = [m["text"] for m in p.get("mention", [])]
        print(f"   Patient {p['patient_id']}: {m_texts[:3]}")
