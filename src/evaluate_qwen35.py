"""Comprehensive benchmark evaluation comparing:
1. Rule Baseline (CPU)
2. GPU SapBERT Enhanced Pipeline
3. GPU SapBERT + Qwen3.5 LLM Patient Association & Negation Filtering

Outputs complete official CHIP2026 metrics for all configurations.
"""

from pathlib import Path
import os
import sys
import re
import json
from typing import Any, Dict, List, Set, Tuple
import torch

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from src.data_loader import HPOntology, load_jsonl, save_jsonl
from src.metrics import compute_all_metrics
from src.pipeline_gpu import GPUEnhancedPipeline, format_metrics
from transformers import AutoModelForCausalLM, AutoTokenizer


class Qwen35AssocModel:
    def __init__(self, model_path_or_id: str = "Qwen/Qwen3.5-0.8B", device: str = "cuda"):
        self.device = device
        print(f"Loading Qwen3.5 model ({model_path_or_id}) onto {device}...")
        try:
            from modelscope import snapshot_download
            actual_path = snapshot_download(model_path_or_id)
        except Exception:
            actual_path = model_path_or_id

        self.tokenizer = AutoTokenizer.from_pretrained(actual_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            actual_path,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
        )
        self.model.eval()
        print("Qwen3.5 model loaded successfully.")

    def verify_patient_phenotype(
        self, patient_name: str, phenotype_text: str, context_sentence: str
    ) -> bool:
        """Ask LLM whether the specific phenotype belongs to the target patient."""
        messages = [
            {
                "role": "user",
                "content": (
                    f"Clinical excerpt: {context_sentence}\n"
                    f"Does the patient ({patient_name}) have {phenotype_text}? "
                    "Answer strictly with YES or NO."
                ),
            }
        ]
        text_in = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text_in], return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            gen_ids = self.model.generate(**inputs, max_new_tokens=32, do_sample=False)
            gen_ids = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, gen_ids)]
            response = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0].strip().upper()

        # If answer explicitly starts with NO, reject
        if response.startswith("NO"):
            return False
        return True


def run_llm_prediction(
    eval_docs: List[Dict[str, Any]],
    gpu_pipeline: GPUEnhancedPipeline,
    llm: Qwen35AssocModel,
) -> List[Dict[str, Any]]:
    pred_docs = []
    print(f"Running Qwen3.5 LLM semantic verification across {len(eval_docs)} documents...")

    for i, doc in enumerate(eval_docs):
        # 1. Base GPU SapBERT extraction for Subtask 1
        base_pred = gpu_pipeline.predict_doc(doc, use_sapbert=True)
        pred_entities = base_pred["entities"]
        patients = doc.get("patient", [])

        # Subtask 2: Semantic verification for patient associations
        pred_associations = []
        for p in patients:
            pid = p["patient_id"]
            mentions = [m.get("text", "") for m in p.get("mention", [])]
            patient_name = mentions[0] if mentions else pid

            # Get candidate phenotypes from proximity base
            base_hpos = next((a["phenotype"] for a in base_pred["association"] if a["patient_id"] == pid), [])

            # If document has multiple patients, use LLM to filter false positives
            if len(patients) > 1 and base_hpos:
                verified_hpos = []
                # Check each associated phenotype
                for hid in base_hpos:
                    # Find context sentence for this phenotype
                    ent_match = next((e for e in pred_entities if e["identifier"] == hid), None)
                    if ent_match:
                        # Extract 300 char window around entity
                        off = ent_match["offset"]
                        # Find enclosing text from doc
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
                        if ctx and not llm.verify_patient_phenotype(patient_name, ent_match["text"], ctx):
                            continue
                    verified_hpos.append(hid)
            else:
                verified_hpos = base_hpos

            pred_associations.append({"patient_id": pid, "phenotype": verified_hpos})

        pred_docs.append({
            "pmc_id": doc["pmc_id"],
            "pmid": doc.get("pmid"),
            "entities": pred_entities,
            "association": pred_associations,
        })
        if (i + 1) % 5 == 0 or i == len(eval_docs) - 1:
            print(f"  Processed {i + 1}/{len(eval_docs)} documents...")

    return pred_docs


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_docs", type=int, default=20, help="Number of train docs to evaluate (default: 20)")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3.5-0.8B")
    args = parser.parse_args()

    train_path = repo_root / "data" / "raw" / "PatientPheX-V1-A" / "PatientPheX-train.jsonl"
    obo_path = repo_root / "data" / "raw" / "PatientPheX-V1-A" / "hp.obo"

    docs = load_jsonl(train_path)
    eval_docs = docs[: args.num_docs]
    gold_docs = eval_docs

    print(f"\n=======================================================")
    print(f"=== Evaluating 3 Paradigms on {len(eval_docs)} Benchmark Documents ===")
    print(f"=======================================================\n")

    # Pipeline 1 & 2: Baseline & GPU SapBERT
    pipeline = GPUEnhancedPipeline(str(obo_path), sim_threshold=0.92)

    print("\n--- 1. Evaluating Rule Baseline ---")
    pred_base = [pipeline.predict_doc(d, use_sapbert=False) for d in eval_docs]
    m_base = format_metrics(compute_all_metrics(gold_docs, pred_base))

    print("\n--- 2. Evaluating GPU SapBERT Enhanced Pipeline ---")
    pred_gpu = [pipeline.predict_doc(d, use_sapbert=True) for d in eval_docs]
    m_gpu = format_metrics(compute_all_metrics(gold_docs, pred_gpu))

    # Pipeline 3: + Qwen3.5
    print(f"\n--- 3. Evaluating GPU SapBERT + Qwen3.5 LLM ({args.model_id}) ---")
    llm = Qwen35AssocModel(args.model_id)
    pred_llm = run_llm_prediction(eval_docs, pipeline, llm)
    m_llm = format_metrics(compute_all_metrics(gold_docs, pred_llm))

    # Print 3-way side-by-side comparison table
    print("\n" + "=" * 82)
    print(f"=== Final 3-Way Benchmark Comparison ({len(eval_docs)} Documents) ===")
    print(f"{'Metric':<20} | {'1. Baseline':<14} | {'2. GPU SapBERT':<16} | {'3. + Qwen3.5':<14} | {'Total Gain':<10}")
    print("-" * 82)
    metrics_to_show = [
        ("overall_score", "Overall Score"),
        ("s1_mention_f1", "S1 Mention F1"),
        ("s1_mention_p", "S1 Mention P"),
        ("s1_mention_r", "S1 Mention R"),
        ("s1_doc_f1", "S1 Doc F1"),
        ("s1_doc_p", "S1 Doc P"),
        ("s1_doc_r", "S1 Doc R"),
        ("s2_micro_f1", "S2 Micro F1"),
        ("s2_micro_p", "S2 Micro P"),
        ("s2_micro_r", "S2 Micro R"),
        ("s2_macro_f1", "S2 Macro F1"),
        ("s2_macro_p", "S2 Macro P"),
        ("s2_macro_r", "S2 Macro R"),
    ]
    for key, name in metrics_to_show:
        v1 = m_base.get(key, 0.0)
        v2 = m_gpu.get(key, 0.0)
        v3 = m_llm.get(key, 0.0)
        gain = v3 - v1
        print(f"{name:<20} | {v1:<14.4f} | {v2:<16.4f} | {v3:<14.4f} | {gain:+.4f}")
    print("=" * 82)


if __name__ == "__main__":
    main()
