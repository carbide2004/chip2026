"""Enhanced streaming inference using SapBERT + Section-weighted Qwen3-8B semantic verification.
Improves upon v1 by:
1. Enhanced clinical negation detection in Task 1.
2. Section-aware weighting:
   - CASE / CLINICAL / RESULTS sections (85% precision) get fast-tracked.
   - INTRO / DISCUSS / OTHER sections (48-52% precision) are strictly verified by Qwen3-8B
     even for single-patient documents, eliminating background discussion false positives.
"""

import argparse
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Set
import torch

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from src.data_loader import HPOntology, load_jsonl, save_jsonl
from src.pipeline_gpu import GPUEnhancedPipeline
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    obo_path = repo_root / "data" / "raw" / "PatientPheX-V1-A" / "hp.obo"
    input_file = repo_root / "data" / "raw" / "PatientPheX-V1-A" / "PatientPheX-A.jsonl"
    base_model = str(Path.home() / ".cache" / "modelscope" / "models" / "Qwen--Qwen3-8B" / "snapshots" / "master")
    output_file = repo_root / "submissions" / "qwen3_8b_v2_optimized_A.jsonl"

    print("=" * 70, flush=True)
    print("=== CHIP2026 PatientPheX Optimized V2 Submission Pipeline ===", flush=True)
    print("=" * 70, flush=True)

    # 1. Pipeline with enhanced negation
    pipeline = GPUEnhancedPipeline(str(obo_path), sim_threshold=0.92)

    # 2. Load base Qwen3-8B
    print(f"Loading Qwen3-8B from {base_model}...", flush=True)
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

    # Define high-confidence clinical section keywords
    CORE_SECTIONS = {"CASE", "CLINICAL", "REPORT", "PRESENT", "RESULTS", "PATIENT"}

    for doc_idx, doc in enumerate(test_docs):
        pmc_id = doc["pmc_id"]
        pmid = doc.get("pmid")
        patients = doc.get("patient", [])

        # Subtask 1: GPU SapBERT soft-matching + dictionary
        base_pred = pipeline.predict_doc(doc, use_sapbert=True)
        pred_entities = base_pred["entities"]

        # Map entity offset to section type
        ent_section_map = {}
        for ent in pred_entities:
            off = ent["offset"]
            s_type = "OTHER"
            for sec in doc.get("full_text", []):
                sec_off = sec.get("offset", 0)
                sec_len = len(sec.get("text", ""))
                if sec_off <= off < sec_off + sec_len:
                    s_type = (sec.get("section_type") or "OTHER").upper()
                    break
            ent_section_map[ent["offset"]] = s_type

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

            # Get candidate phenotypes from proximity base
            base_hpos = next((a["phenotype"] for a in base_pred["association"] if a["patient_id"] == pid), [])

            if not base_hpos:
                pred_associations.append({"patient_id": pid, "phenotype": []})
                print("done -> 0 candidates.", flush=True)
                continue

            verified_hpos = []
            for hid in base_hpos:
                ent_match = next((e for e in pred_entities if e["identifier"] == hid), None)
                if not ent_match:
                    continue

                off = ent_match["offset"]
                sec_type = ent_section_map.get(off, "OTHER")
                is_core = any(cs in sec_type for cs in CORE_SECTIONS)

                # If single patient AND located in core clinical section (CASE/RESULTS), fast-path accept
                if len(patients) == 1 and is_core:
                    verified_hpos.append(hid)
                    continue

                # Otherwise (multi-patient OR in background sections INTRO/DISCUSS), verify with Qwen3-8B
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
                        f"Question: Does the patient ({patient_name}) have {ent_match['text']}?\n"
                        f"<think>\nAnalyze whether {ent_match['text']} is explicitly confirmed for {patient_name}.\n</think>\n"
                        "Answer (YES or NO):"
                    )
                    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
                    with torch.no_grad():
                        gen_ids = model.generate(**inputs, max_new_tokens=5, do_sample=False)
                        gen_ids = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, gen_ids)]
                        resp = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0].strip().upper()

                    # Only exclude if model clearly denies it
                    if "NO" in resp and "YES" not in resp:
                        continue

                verified_hpos.append(hid)

            pred_associations.append({"patient_id": pid, "phenotype": verified_hpos})
            print(f"done -> {len(verified_hpos)} / {len(base_hpos)} HPO IDs retained.", flush=True)

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
    print(f"Optimized V2 submission saved to: {output_file}", flush=True)
    print(f"Total time elapsed: {time.time() - t0:.1f} seconds.", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
