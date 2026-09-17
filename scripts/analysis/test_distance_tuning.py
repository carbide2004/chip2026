"""Evaluate dynamic paragraph/sentence-level windowing vs 1500 char distance.
"""

import sys
from pathlib import Path
import re

repo_root = Path(r"E:\Workplace_Tanhaowen\chip2026-patientphex")
sys.path.insert(0, str(repo_root))

from src.data_loader import load_jsonl, HPOntology
from src.pipeline_gpu import GPUEnhancedPipeline, format_metrics
from src.metrics import compute_all_metrics

docs = load_jsonl(repo_root / "data" / "raw" / "PatientPheX-V1-A" / "PatientPheX-train.jsonl")
eval_docs = docs[:20]
obo = repo_root / "data" / "raw" / "PatientPheX-V1-A" / "hp.obo"

pipeline = GPUEnhancedPipeline(str(obo), sim_threshold=0.92)

# Compare different distance windows in multi-patient docs: 1500 vs 500 vs 300
for window in [1500, 800, 400]:
    out_docs = []
    for d in eval_docs:
        p_res = pipeline.predict_doc(d, use_sapbert=True)
        ents = [e for e in p_res["entities"] if e["identifier"] != "-1" and e.get("note") is None]
        patients = d.get("patient", [])
        
        assocs = []
        for p in patients:
            pid = p["patient_id"]
            p_offs = [m["offset"] for m in p.get("mention", [])]
            
            p_hpos = []
            seen = set()
            for e in ents:
                # If only 1 patient, associate all; if multi-patient, use window
                is_close = any(abs(e["offset"] - poff) < window for poff in p_offs)
                if len(patients) == 1 or is_close:
                    if e["identifier"] not in seen:
                        seen.add(e["identifier"])
                        p_hpos.append(e["identifier"])
            assocs.append({"patient_id": pid, "phenotype": p_hpos})
            
        p_res["association"] = assocs
        out_docs.append(p_res)
        
    m = format_metrics(compute_all_metrics(eval_docs, out_docs))
    print(f"Window {window:4d}: Overall={m['overall_score']:.4f} | S2 Micro={m['s2_micro_f1']:.4f} (P={m['s2_micro_p']:.4f}, R={m['s2_micro_r']:.4f}) | S2 Macro={m['s2_macro_f1']:.4f} (P={m['s2_macro_p']:.4f}, R={m['s2_macro_r']:.4f})")
