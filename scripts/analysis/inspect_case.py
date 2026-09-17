import sys
import json
from pathlib import Path

repo_root = Path(r"E:\Workplace_Tanhaowen\chip2026-patientphex")
sys.path.insert(0, str(repo_root))

from src.data_loader import load_jsonl

docs = load_jsonl(repo_root / "data" / "raw" / "PatientPheX-V1-A" / "PatientPheX-train.jsonl")

# Let's inspect PMC11196099: P1 (38-year-old woman) vs P2 (Her 65-year old mother)
d = next(d for d in docs if d["pmc_id"] == "11196099")
gold_assoc = {a["patient_id"]: a["phenotype"] for a in d["association"]}

print("=== Deep Dive into PMC11196099 ===")
print("Patient P1 mentions:", [m["text"] for m in d["patient"][0]["mention"]])
print("Patient P2 mentions:", [m["text"] for m in d["patient"][1]["mention"]])
print("\nGold P1 phenotypes (count %d):" % len(gold_assoc["P1"]), gold_assoc["P1"])
print("\nGold P2 phenotypes (count %d):" % len(gold_assoc["P2"]), gold_assoc["P2"])

# Find text sections mentioning mother
for sec in d.get("full_text", []):
    txt = sec.get("text", "")
    if "mother" in txt.lower():
        print("\n--- Section mentioning mother (%s) ---" % sec.get("section_type"))
        print(txt[:800].replace("\n", " "))
