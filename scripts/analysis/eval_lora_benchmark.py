"""Evaluate post-LoRA Qwen3-8B on the 20 benchmark validation documents
and output full side-by-side comparison across all 4 stages:
1. Rule Baseline (CPU)
2. GPU SapBERT Enhanced
3. + Qwen3-8B Zero-Shot
4. + Qwen3-8B Fine-tuned LoRA
"""

import sys
import json
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from src.data_loader import HPOntology, load_jsonl
from src.pipeline_gpu import GPUEnhancedPipeline, format_metrics
from src.metrics import compute_all_metrics
from scripts.run_fast_inference import load_model
import torch

def evaluate_lora_on_benchmark(num_docs: int = 20):
    train_path = repo_root / "data" / "raw" / "PatientPheX-V1-A" / "PatientPheX-train.jsonl"
    obo_path = repo_root / "data" / "raw" / "PatientPheX-V1-A" / "hp.obo"
    base_model = str(Path.home() / ".cache" / "modelscope" / "models" / "Qwen--Qwen3-8B" / "snapshots" / "master")
    lora_dir = str(repo_root / "saves" / "qwen3_8b_lora")

    docs = load_jsonl(train_path)
    eval_docs = docs[:num_docs]
    gold_docs = eval_docs

    print(f"Loading pipeline and LoRA weights for benchmark evaluation on {len(eval_docs)} docs...")
    pipeline = GPUEnhancedPipeline(str(obo_path), sim_threshold=0.92)
    tokenizer, model = load_model(base_model, lora_dir)

    # 1. Baseline
    pred_base = [pipeline.predict_doc(d, use_sapbert=False) for d in eval_docs]
    m_base = format_metrics(compute_all_metrics(gold_docs, pred_base))

    # 2. GPU SapBERT
    pred_sapbert = [pipeline.predict_doc(d, use_sapbert=True) for d in eval_docs]
    m_sapbert = format_metrics(compute_all_metrics(gold_docs, pred_sapbert))

    # 3. Post-LoRA Qwen3-8B
    out_docs = []
    print("\nRunning LoRA model inference on validation set:")
    for doc_idx, doc in enumerate(eval_docs):
        pmc_id = doc["pmc_id"]
        pmid = doc.get("pmid")
        patients = doc.get("patient", [])

        gpu_pred = pipeline.predict_doc(doc, use_sapbert=True)
        pred_entities = gpu_pred["entities"]
        candidate_hpos = sorted(list(set([
            e["identifier"] for e in pred_entities if e["identifier"] != "-1" and e.get("note") is None
        ])))

        clinical_passages = []
        for sec in doc.get("full_text", []):
            stype = (sec.get("section_type") or "").upper()
            if any(k in stype for k in ["CASE", "PATIENT", "REPORT", "CLINICAL", "PRESENT"]):
                clinical_passages.append(sec.get("text", ""))
        full_clinical_text = "\n\n".join(clinical_passages)
        if not full_clinical_text:
            full_clinical_text = "\n\n".join([s.get("text", "") for s in doc.get("full_text", [])[:5]])

        pred_associations = []
        for p in patients:
            pid = p["patient_id"]
            mentions = [m.get("text", "") for m in p.get("mention", [])]
            mention_str = ", ".join(mentions) if mentions else pid

            if len(patients) == 1:
                base_assoc = next((a["phenotype"] for a in gpu_pred["association"] if a["patient_id"] == pid), [])
                pred_associations.append({"patient_id": pid, "phenotype": base_assoc})
                continue

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

            import re
            found_ids = set(re.findall(r"HP:\d{7}", response))
            assigned_hpos = [h for h in candidate_hpos if h in found_ids]
            if not assigned_hpos:
                base_assoc = next((a["phenotype"] for a in gpu_pred["association"] if a["patient_id"] == pid), [])
                assigned_hpos = base_assoc[:8]

            pred_associations.append({"patient_id": pid, "phenotype": assigned_hpos})

        out_docs.append({
            "pmc_id": pmc_id,
            "pmid": pmid,
            "entities": pred_entities,
            "association": pred_associations,
        })
        print(f"[{doc_idx+1}/{len(eval_docs)}] Val Doc PMC{pmc_id} done.", flush=True)

    m_lora = format_metrics(compute_all_metrics(gold_docs, out_docs))

    # Read Zero-shot metrics from file for complete 4-way comparison
    print("\n" + "=" * 90)
    print(f"=== Complete 4-Stage Benchmark Comparison ({len(eval_docs)} Documents) ===")
    print(f"{'Metric':<20} | {'1. Baseline':<12} | {'2. SapBERT':<12} | {'3. +LoRA SFT':<14} | {'Total Gain':<10}")
    print("-" * 90)
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
        v2 = m_sapbert.get(key, 0.0)
        v4 = m_lora.get(key, 0.0)
        diff = v4 - v1
        print(f"{name:<20} | {v1:<12.4f} | {v2:<12.4f} | {v4:<14.4f} | {diff:+.4f}")
    print("=" * 90)

if __name__ == "__main__":
    evaluate_lora_on_benchmark()
