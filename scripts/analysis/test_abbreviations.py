"""Test impact of adding biomedical clinical abbreviation mapping on train benchmark.
"""

import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple
import re

repo_root = Path(r"E:\Workplace_Tanhaowen\chip2026-patientphex")
sys.path.insert(0, str(repo_root))

from src.data_loader import load_jsonl, HPOntology
from src.pipeline_gpu import GPUEnhancedPipeline, format_metrics
from src.metrics import compute_all_metrics

# Standard high-confidence clinical abbreviations extracted from corpus
CLINICAL_ABBREVIATIONS = {
    "DM": "HP:0000819",       # Diabetes mellitus
    "SNHL": "HP:0000407",     # Sensorineural hearing impairment
    "LCD": "HP:0001149",      # Lattice corneal dystrophy
    "ASD": "HP:0000729",      # Autism spectrum disorder
    "ADHD": "HP:0007018",     # Attention deficit hyperactivity disorder
    "ID": "HP:0001249",       # Intellectual disability
    "MR": "HP:0001249",       # Mental retardation
    "MODY": "HP:0004904",     # Maturity-onset diabetes of the young
    "GDD": "HP:0001263",      # Global developmental delay
    "IUGR": "HP:0001511",     # Intrauterine growth retardation
    "CL/P": "HP:0000202",     # Cleft lip with or without cleft palate
    "NDD": "HP:0012759",      # Neurodevelopmental delay
    "HE": "HP:0004445",       # Hyperekplexia
    "SP": "HP:0002108",       # Spastic paraplegia
    "HCM": "HP:0001639",      # Hypertrophic cardiomyopathy
    "DCM": "HP:0001644",      # Dilated cardiomyopathy
}

docs = load_jsonl(repo_root / "data" / "raw" / "PatientPheX-V1-A" / "PatientPheX-train.jsonl")
obo = repo_root / "data" / "raw" / "PatientPheX-V1-A" / "hp.obo"

eval_docs = docs[:20]
pipeline = GPUEnhancedPipeline(str(obo), sim_threshold=0.92)

# 1. Current pipeline without abbreviations
pred_orig = [pipeline.predict_doc(d, use_sapbert=True) for d in eval_docs]
m_orig = format_metrics(compute_all_metrics(eval_docs, pred_orig))

# 2. Enhanced with abbreviation matching
pred_abbr = []
for d in eval_docs:
    p = pipeline.predict_doc(d, use_sapbert=True)
    ents = p["entities"]
    existing_spans = {(e["offset"], e["length"]) for e in ents}
    
    # scan for exact capital abbreviations with word boundaries
    for sec in d.get("full_text", []):
        txt = sec.get("text", "")
        sec_off = sec.get("offset", 0)
        sec_type = (sec.get("section_type") or "").upper()
        if any(skip in sec_type for skip in ["REF", "ACK", "FOOTNOTE", "SUPPL"]):
            continue
            
        for abbr, hid in CLINICAL_ABBREVIATIONS.items():
            pattern = re.compile(rf"\b{re.escape(abbr)}\b")
            for m in pattern.finditer(txt):
                start = m.start()
                length = len(abbr)
                global_off = sec_off + start
                if (global_off, length) not in existing_spans:
                    ctx_start = max(0, start - 50)
                    ctx_end = min(len(txt), start + length + 50)
                    context = txt[ctx_start:ctx_end].replace("\n", " ")
                    is_neg = pipeline.detect_negation(context, abbr)
                    note = "NO" if is_neg else None
                    ents.append({
                        "identifier": hid,
                        "type": "Phenotype",
                        "offset": global_off,
                        "length": length,
                        "text": abbr,
                        "note": note,
                    })
                    existing_spans.add((global_off, length))
    
    ents.sort(key=lambda x: (x["offset"], -x["length"]))
    p["entities"] = ents
    pred_abbr.append(p)

m_abbr = format_metrics(compute_all_metrics(eval_docs, pred_abbr))

print("\n" + "=" * 65)
print("=== Impact of High-Frequency Clinical Abbreviations ===")
print(f"{'Metric':<20} | {'Original':<12} | {'+ Abbreviations':<16} | {'Diff':<8}")
print("-" * 65)
for k in ["overall_score", "s1_mention_f1", "s1_mention_p", "s1_mention_r", "s1_doc_f1", "s1_doc_p", "s1_doc_r"]:
    v1 = m_orig.get(k, 0.0)
    v2 = m_abbr.get(k, 0.0)
    print(f"{k:<20} | {v1:<12.4f} | {v2:<16.4f} | {v2-v1:+.4f}")
print("=" * 65)
