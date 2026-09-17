import sys
import json
from pathlib import Path

repo_root = Path(r"E:\Workplace_Tanhaowen\chip2026-patientphex")
sys.path.insert(0, str(repo_root))

from src.data_loader import load_jsonl, HPOntology
from src.pipeline_gpu import GPUEnhancedPipeline

docs = load_jsonl(repo_root / "data" / "raw" / "PatientPheX-V1-A" / "PatientPheX-train.jsonl")
d = next(d for d in docs if d["pmc_id"] == "11196099")
obo = repo_root / "data" / "raw" / "PatientPheX-V1-A" / "hp.obo"

pipeline = GPUEnhancedPipeline(str(obo), sim_threshold=0.92)
pred = pipeline.predict_doc(d, use_sapbert=True)

print("Gold P2 (mother) phenotypes:", d["association"][1]["phenotype"])
print("Pipeline base associated for P2:", [a["phenotype"] for a in pred["association"] if a["patient_id"] == "P2"][0])
