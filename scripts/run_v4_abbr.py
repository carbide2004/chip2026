"""Pipeline V4:
1. Retains the high-scoring Qwen3-8B semantic reasoning for Task 2 (Macro F1 0.4837).
2. Upgrades Task 1 with clinical capital abbreviation boundary matching (SNHL, DM, LCD, ASD, etc.),
   targeting the 50% score weight directly to break past 0.61+.
"""

import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Set, Tuple
import torch

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from src.data_loader import HPOntology, load_jsonl, save_jsonl
from src.pipeline_gpu import GPUEnhancedPipeline
from transformers import AutoModelForCausalLM, AutoTokenizer

# High-frequency biomedical abbreviations verified in genetics cases
CLINICAL_ABBREVIATIONS = {
    "SNHL": "HP:0000407",     # Sensorineural hearing impairment
    "DM": "HP:0000819",       # Diabetes mellitus
    "LCD": "HP:0001149",      # Lattice corneal dystrophy
    "ASD": "HP:0000729",      # Autism spectrum disorder
    "ADHD": "HP:0007018",     # Attention deficit hyperactivity disorder
    "ID": "HP:0001249",       # Intellectual disability
    "MR": "HP:0001249",       # Mental retardation
    "GDD": "HP:0001263",      # Global developmental delay
    "MODY": "HP:0004904",     # Maturity-onset diabetes of the young
    "CL/P": "HP:0000202",     # Cleft lip with or without cleft palate
    "NDD": "HP:0012759",      # Neurodevelopmental delay
    "HE": "HP:0004445",       # Hyperekplexia
    "SP": "HP:0002108",       # Spastic paraplegia
    "HCM": "HP:0001639",      # Hypertrophic cardiomyopathy
    "DCM": "HP:0001644",      # Dilated cardiomyopathy
    "IUGR": "HP:0001511",     # Intrauterine growth retardation
}


def add_abbreviations_to_doc(doc: Dict[str, Any], pipeline: GPUEnhancedPipeline, existing_entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Scan document text for strict capital abbreviation matches with boundary checks."""
    ents = list(existing_entities)
    existing_spans = {(e["offset"], e["length"]) for e in ents}

    for sec in doc.get("full_text", []):
        txt = sec.get("text", "")
        sec_off = sec.get("offset", 0)
        sec_type = (sec.get("section_type") or "").upper()
        if any(skip in sec_type for skip in ["REF", "ACK", "FOOTNOTE", "SUPPL", "AUTHOR"]):
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
    return ents


def main():
    obo_path = repo_root / "data" / "raw" / "PatientPheX-V1-A" / "hp.obo"
    input_file = repo_root / "data" / "raw" / "PatientPheX-V1-A" / "PatientPheX-A.jsonl"
    base_model = str(Path.home() / ".cache" / "modelscope" / "models" / "Qwen--Qwen3-8B" / "snapshots" / "master")
    output_file = repo_root / "submissions" / "qwen3_8b_v4_abbr_A.jsonl"

    print("=" * 70, flush=True)
    print("=== CHIP2026 PatientPheX V4 Pipeline (Task 1 Abbr + Task 2 Balanced) ===", flush=True)
    print("=" * 70, flush=True)

    pipeline = GPUEnhancedPipeline(str(obo_path), sim_threshold=0.92)

    print(f"Loading base Qwen3-8B from {base_model}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print("Model loaded successfully.\n", flush=True)

    test_docs = load_jsonl(input_file)
    total_patients = sum(len(d.get("patient", [])) for d in test_docs)
    print(f"Loaded {len(test_docs)} documents, {total_patients} total patients.", flush=True)

    out_docs = []
    patient_global_idx = 0
    t0 = time.time()

    for doc_idx, doc in enumerate(test_docs):
        pmc_id = doc["pmc_id"]
        pmid = doc.get("pmid")
        patients = doc.get("patient", [])

        # Subtask 1: SapBERT soft-matching + Dictionary + Clinical Abbrs
        base_pred = pipeline.predict_doc(doc, use_sapbert=True)
        pred_entities = add_abbreviations_to_doc(doc, pipeline, base_pred["entities"])

        # Subtask 2: Association
        pred_associations = []
        for p in patients:
            patient_global_idx += 1
            pid = p["patient_id"]
            mentions = [m.get("text", "") for m in p.get("mention", [])]
            patient_name = mentions[0] if mentions else pid

            print(
                f"[{patient_global_idx:02d}/{total_patients}] Doc {doc_idx+1}/{len(test_docs)} (PMC{pmc_id}) Patient '{pid}'... ",
                end="",
                flush=True,
            )

            # Build candidates within 1500 chars of patient mentions
            patient_offsets = [m["offset"] for m in p.get("mention", [])]
            candidate_hpos = []
            seen = set()
            for ent in pred_entities:
                if ent["identifier"] == "-1" or ent.get("note") is not None:
                    continue
                is_close = any(abs(ent["offset"] - poff) < 1500 for poff in patient_offsets)
                if len(patients) == 1 or is_close:
                    if ent["identifier"] not in seen:
                        seen.add(ent["identifier"])
                        candidate_hpos.append(ent["identifier"])

            # Single patient document: fast-path preserve
            if len(patients) == 1 or not candidate_hpos:
                pred_associations.append({"patient_id": pid, "phenotype": candidate_hpos})
                print(f"[Single-P Fast-path] -> {len(candidate_hpos)} HPO IDs assigned.", flush=True)
                continue

            # Multi-patient document: Qwen3-8B semantic verification
            verified_hpos = []
            for hid in candidate_hpos:
                ent_match = next((e for e in pred_entities if e["identifier"] == hid), None)
                if ent_match:
                    off = ent_match["offset"]
                    ctx = ""
                    for sec in doc.get("full_text", []):
                        sec_off = sec.get("offset", 0)
                        sec_len = len(sec.get("text", ""))
                        if sec_off <= off < sec_off + sec_len:
                            rel_pos = off - sec_off
                            start_c = max(0, rel_pos - 150)
                            end_c = min(sec_len, rel_pos + 150)
                            ctx = sec.get("text", "")[start_c:end_c].replace("\n", " ").strip()
                            break

                    if ctx:
                        prompt = (
                            f"Context: {ctx}\n"
                            f"Target Subject: {patient_name}\n"
                            f"Question: Does the target subject ({patient_name}) have {ent_match['text']}? "
                            f"If it belongs to another relative or is absent, answer NO. If confirmed or likely for target subject, answer YES.\n"
                            f"<think>\nEvaluate if {ent_match['text']} belongs specifically to {patient_name}.\n</think>\n"
                            "Answer (YES or NO):"
                        )
                        inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
                        with torch.no_grad():
                            gen_ids = model.generate(**inputs, max_new_tokens=5, do_sample=False)
                            gen_ids = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, gen_ids)]
                            resp = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0].strip().upper()

                        if "NO" in resp and "YES" not in resp:
                            continue

                verified_hpos.append(hid)

            pred_associations.append({"patient_id": pid, "phenotype": verified_hpos})
            print(f"done -> {len(verified_hpos)} / {len(candidate_hpos)} HPO IDs retained.", flush=True)

        out_docs.append({
            "pmc_id": pmc_id,
            "pmid": pmid,
            "entities": pred_entities,
            "association": pred_associations,
        })

    output_file.parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(out_docs, str(output_file))

    print("\n" + "=" * 70, flush=True)
    print(f"SUCCESS! All {len(out_docs)} documents processed.", flush=True)
    print(f"Submission saved to: {output_file}", flush=True)
    print(f"Total time elapsed: {time.time() - t0:.1f} seconds.", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
