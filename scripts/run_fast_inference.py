"""Fast and accurate inference combining GPU SapBERT with Qwen3-8B semantic verification.
Features:
- Real-time streaming progress printing for every patient.
- Prompt format strictly aligned with SFT training.
- Fallback guard to ensure recall.
"""

import argparse
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Set

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from src.data_loader import HPOntology, load_jsonl, save_jsonl
from src.pipeline_gpu import GPUEnhancedPipeline


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model",
        type=str,
        default=str(Path.home() / ".cache" / "modelscope" / "models" / "Qwen--Qwen3-8B" / "snapshots" / "master"),
    )
    parser.add_argument("--lora_dir", type=str, default=str(repo_root / "saves" / "qwen3_8b_lora"))
    parser.add_argument(
        "--input_file",
        type=str,
        default=str(repo_root / "data" / "raw" / "PatientPheX-V1-A" / "PatientPheX-A.jsonl"),
    )
    parser.add_argument(
        "--obo_file",
        type=str,
        default=str(repo_root / "data" / "raw" / "PatientPheX-V1-A" / "hp.obo"),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(repo_root / "submissions" / "qwen3_8b_final_A.jsonl"),
    )
    return parser.parse_args()


def load_model(base_model: str, lora_dir: str = None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print(f"Loading Qwen3-8B base model from {base_model}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    if lora_dir and os.path.exists(lora_dir):
        print(f"Loading LoRA adapter from {lora_dir}...", flush=True)
        model = PeftModel.from_pretrained(model, lora_dir)
    model.eval()
    print("Model successfully loaded and ready for inference.", flush=True)
    return tokenizer, model


def main():
    args = parse_args()
    print("=" * 60, flush=True)
    print("=== CHIP2026 PatientPheX Final A-Test Pipeline ===", flush=True)
    print("=" * 60, flush=True)

    # 1. Initialize GPU SapBERT pipeline
    pipeline = GPUEnhancedPipeline(args.obo_file, sim_threshold=0.92)

    # 2. Load model
    import torch
    tokenizer, model = load_model(args.base_model, args.lora_dir)

    test_docs = load_jsonl(args.input_file)
    total_patients = sum(len(d.get("patient", [])) for d in test_docs)
    print(f"\nLoaded {len(test_docs)} documents, {total_patients} total patients to process.", flush=True)
    print("Starting streaming inference:\n", flush=True)

    out_docs = []
    patient_global_idx = 0
    start_time = time.time()

    for doc_idx, doc in enumerate(test_docs):
        pmc_id = doc["pmc_id"]
        pmid = doc.get("pmid")
        patients = doc.get("patient", [])

        # Subtask 1: GPU SapBERT soft-matching + dictionary
        gpu_pred = pipeline.predict_doc(doc, use_sapbert=True)
        pred_entities = gpu_pred["entities"]

        candidate_hpos = sorted(list(set([
            e["identifier"] for e in pred_entities if e["identifier"] != "-1" and e.get("note") is None
        ])))

        # Extract primary clinical passage
        clinical_passages = []
        for sec in doc.get("full_text", []):
            stype = (sec.get("section_type") or "").upper()
            if any(k in stype for k in ["CASE", "PATIENT", "REPORT", "CLINICAL", "PRESENT"]):
                clinical_passages.append(sec.get("text", ""))
        full_clinical_text = "\n\n".join(clinical_passages)
        if not full_clinical_text:
            full_clinical_text = "\n\n".join([s.get("text", "") for s in doc.get("full_text", [])[:5]])

        # Subtask 2: Query Qwen3 for each patient
        pred_associations = []
        for p in patients:
            patient_global_idx += 1
            pid = p["patient_id"]
            mentions = [m.get("text", "") for m in p.get("mention", [])]
            mention_str = ", ".join(mentions) if mentions else pid

            print(
                f"[{patient_global_idx:02d}/{total_patients}] Doc {doc_idx+1}/{len(test_docs)} (PMC{pmc_id}) - Patient '{pid}'... ",
                end="",
                flush=True,
            )

            # Fast path: Single patient document associates all non-negated phenotypes in clinical text
            if len(patients) == 1:
                base_assoc = next((a["phenotype"] for a in gpu_pred["association"] if a["patient_id"] == pid), [])
                pred_associations.append({"patient_id": pid, "phenotype": base_assoc})
                print(f"[Single Patient Fast-path] -> {len(base_assoc)} phenotypes assigned.", flush=True)
                continue

            # Multi-patient document: invoke LoRA model for precise patient association
            instruction = (
                f"You are an expert clinical geneticist. Below is the clinical presentation from a biomedical article "
                f"and information about patient '{pid}' (mentioned in text as: {mention_str}). "
                f"From the provided list of candidate HPO phenotype IDs present in the article, determine which phenotypes "
                f"are specifically confirmed and belong to this patient. Exclude phenotypes belonging to relatives, "
                f"negated symptoms, or healthy traits. Output your answer as a JSON array of confirmed HPO IDs."
            )
            user_input = (
                f"Candidate HPO IDs: {json.dumps(candidate_hpos)}\n\n"
                f"Clinical Text:\n{full_clinical_text[:2800]}"
            )

            conv = [
                {"role": "system", "content": instruction},
                {"role": "user", "content": user_input},
            ]
            text_in = tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer([text_in], return_tensors="pt").to(model.device)

            with torch.no_grad():
                gen_ids = model.generate(**inputs, max_new_tokens=160, do_sample=False)
                gen_ids = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, gen_ids)]
                response = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()

            # Parse HPO IDs using regex
            found_ids = set(re.findall(r"HP:\d{7}", response))
            assigned_hpos = [h for h in candidate_hpos if h in found_ids]

            # Ensemble fallback: If model produced 0 assoc for target proband, keep proximal entities
            if not assigned_hpos:
                base_assoc = next((a["phenotype"] for a in gpu_pred["association"] if a["patient_id"] == pid), [])
                assigned_hpos = base_assoc[:8]

            pred_associations.append({"patient_id": pid, "phenotype": assigned_hpos})
            elapsed = time.time() - start_time
            print(f"Done -> {len(assigned_hpos)} HPO IDs assigned (elapsed: {elapsed:.1f}s)", flush=True)

        out_docs.append({
            "pmc_id": pmc_id,
            "pmid": pmid,
            "entities": pred_entities,
            "association": pred_associations,
        })

    # Save output
    out_file = Path(args.output)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(out_docs, str(out_file))

    print("\n" + "=" * 60, flush=True)
    print(f"SUCCESS! All {len(out_docs)} documents processed.", flush=True)
    print(f"Final A-test submission saved to: {out_file}", flush=True)
    print(f"Total time elapsed: {time.time() - start_time:.1f} seconds.", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
