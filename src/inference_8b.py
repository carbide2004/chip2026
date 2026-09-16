"""Run local / GPU inference using Qwen3.5 (e.g. Qwen/Qwen3.5-9B or Qwen/Qwen3.5-4B) with SapBERT.

Usage:
  python src/inference_8b.py --base_model Qwen/Qwen3.5-9B --output submissions/qwen35_pred_A.jsonl
"""

import argparse
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Set

# Default to domestic mirrors for speed
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from src.data_loader import HPOntology, load_jsonl, save_jsonl
from src.pipeline_gpu import GPUEnhancedPipeline


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3.5-9B")
    parser.add_argument("--lora_dir", type=str, default=None)
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
        default=str(repo_root / "submissions" / "qwen35_pred_A.jsonl"),
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use_modelscope", action="store_true", default=True, help="Download model via ModelScope")
    return parser.parse_args()


def load_llm_model(model_name: str, lora_dir: str = None, device: str = "cuda"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    actual_model_path = model_name
    try:
        from modelscope import snapshot_download
        print(f"Checking / downloading {model_name} via ModelScope...")
        actual_model_path = snapshot_download(model_name)
        print(f"Model path: {actual_model_path}")
    except Exception as me:
        print(f"ModelScope fallback to HF: {me}")
        actual_model_path = model_name

    tokenizer = AutoTokenizer.from_pretrained(actual_model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        actual_model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    if lora_dir:
        print(f"Loading LoRA weights from {lora_dir}...")
        model = PeftModel.from_pretrained(model, lora_dir)
    model.eval()
    return tokenizer, model


def main():
    args = parse_args()
    print(f"Initializing GPU SapBERT Pipeline with {args.obo_file}...")
    pipeline = GPUEnhancedPipeline(args.obo_file, sim_threshold=0.92)

    print(f"Loading LLM pipeline: {args.base_model} (LoRA: {args.lora_dir})...")
    tokenizer, model = load_llm_model(args.base_model, args.lora_dir, args.device)
    import torch

    test_docs = load_jsonl(args.input_file)
    print(f"Predicting {len(test_docs)} documents...", flush=True)
    out_docs = []

    for doc_idx, doc in enumerate(test_docs):
        pmc_id = doc["pmc_id"]
        pmid = doc.get("pmid")
        patients = doc.get("patient", [])
        print(f"[{doc_idx + 1}/{len(test_docs)}] Processing PMC{pmc_id} ({len(patients)} patients)...", flush=True)

        # Subtask 1: GPU SapBERT soft-matching + dictionary
        gpu_pred = pipeline.predict_doc(doc, use_sapbert=True)
        pred_entities = gpu_pred["entities"]

        candidate_hpos = sorted(list(set([
            e["identifier"] for e in pred_entities if e["identifier"] != "-1" and e.get("note") is None
        ])))

        # Extract clinical presentation
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
        for p_idx, p in enumerate(patients):
            pid = p["patient_id"]
            mentions = [m.get("text", "") for m in p.get("mention", [])]
            print(f"  -> Patient [{p_idx + 1}/{len(patients)}] '{pid}' ({len(candidate_hpos)} cand HPOs)...", end="", flush=True)

            if not candidate_hpos:
                pred_associations.append({"patient_id": pid, "phenotype": []})
                print(" done (0 cand).", flush=True)
                continue

            prompt = (
                f"You are an expert clinical geneticist. Determine which of the candidate HPO phenotype IDs "
                f"specifically belong to patient '{pid}' (mentioned as: {', '.join(mentions)}) based on the clinical text.\n\n"
                f"Candidate HPO IDs: {json.dumps(candidate_hpos)}\n\n"
                f"Clinical Text:\n{full_clinical_text[:3500]}\n\n"
                "Output ONLY a JSON list of confirmed HPO IDs, e.g. [\"HP:0001234\", ...]:"
            )

            messages = [{"role": "user", "content": prompt}]
            text_in = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer([text_in], return_tensors="pt").to(model.device)

            with torch.no_grad():
                gen_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)
                gen_ids = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, gen_ids)]
                response = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()

            # Parse JSON from LLM response
            m = re.search(r"\[.*?\]", response, re.DOTALL)
            assigned_hpos = []
            if m:
                try:
                    assigned_hpos = json.loads(m.group(0))
                except Exception:
                    assigned_hpos = []

            # Filter to valid candidates
            assigned_hpos = [h for h in assigned_hpos if h in candidate_hpos]

            pred_associations.append({
                "patient_id": pid,
                "phenotype": assigned_hpos
            })
            print(f" done ({len(assigned_hpos)} assoc).", flush=True)

        out_docs.append({
            "pmc_id": pmc_id,
            "pmid": pmid,
            "entities": pred_entities,
            "association": pred_associations,
        })

    save_jsonl(out_docs, args.output)
    print(f"Predictions successfully saved to {args.output}", flush=True)


if __name__ == "__main__":
    main()
