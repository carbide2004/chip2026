import sys
from pathlib import Path

repo_root = Path(r"E:\Workplace_Tanhaowen\chip2026-patientphex")
sys.path.insert(0, str(repo_root))

from src.data_loader import load_jsonl

docs = load_jsonl(repo_root / "data" / "raw" / "PatientPheX-V1-A" / "PatientPheX-train.jsonl")
d = next(d for d in docs if d["pmc_id"] == "11196099")

for sec in d.get("full_text", []):
    txt = sec.get("text", "")
    idx = txt.lower().find("mother")
    if idx != -1:
        print(f"--- Section {sec.get('section_type')} offset {sec.get('offset')} ---")
        start = max(0, idx - 100)
        end = min(len(txt), idx + 400)
        print(txt[start:end].replace("\n", " "))
        print()
