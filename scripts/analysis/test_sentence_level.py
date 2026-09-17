import sys
from pathlib import Path
import re

repo_root = Path(r"E:\Workplace_Tanhaowen\chip2026-patientphex")
sys.path.insert(0, str(repo_root))

from src.data_loader import load_jsonl, HPOntology
from src.pipeline_gpu import GPUEnhancedPipeline

docs = load_jsonl(repo_root / "data" / "raw" / "PatientPheX-V1-A" / "PatientPheX-train.jsonl")
d = next(d for d in docs if d["pmc_id"] == "11196099")
obo = repo_root / "data" / "raw" / "PatientPheX-V1-A" / "hp.obo"

pipeline = GPUEnhancedPipeline(str(obo), sim_threshold=0.92)
pred = pipeline.predict_doc(d, use_sapbert=True)

# Full text of doc
full_text = ""
for sec in d.get("full_text", []):
    full_text += sec.get("text", "") + "\n"

# Split text into sentences using simple regex
sentences = list(re.finditer(r'[^.!?]+[.!?]', full_text))

m_off = d["patient"][1]["mention"][0]["offset"]
# find sentence containing mother
mother_sent = next((s.group(0) for s in sentences if s.start() <= m_off < s.end()), "")
print("Sentence containing mother:")
print(repr(mother_sent.strip()))

# Phenotypes in that same sentence:
print("\nPhenotypes in that exact sentence:")
for s in sentences:
    if s.start() <= m_off < s.end():
        for e in pred["entities"]:
            if s.start() <= e["offset"] < s.end():
                print("  ->", e["text"], e["identifier"])
