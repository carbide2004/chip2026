"""Run local / GPU inference using fine-tuned 8B model or zero-shot prompt.

Usage:
  python src/inference_8b.py --base_model Qwen/Qwen2.5-7B-Instruct --lora_dir saves/qwen-patient-lora --output submissions/8b_pred_A.jsonl
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Set

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from src.data_loader import HPOntology, load_jsonl, save_jsonl
from src.dictionary_extractor import DictionaryExtractor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--lora_dir", type=str, default=None)
    parser.add_argument("--input_file", type=str, default=str(repo_root / "data" / "raw" / "PatientPheX-V1-A" / "PatientPheX-A.jsonl"))
    parser.add_argument("--obo_file", type=str, default=str(repo_root / "data" / "raw" / "PatientPheX-V1-A" / "hp.obo"))
    parser.add_argument("--output", type=str, default=str(repo_root / "submissions" / "8b_pred_A.jsonl"))
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Loading HPO ontology from {args.obo_file}...")
    hpo = HPOntology(args.obo_file)
    extractor = DictionaryExtractor(hpo)

    print(f"Loading LLM pipeline: {args.base_model} (LoRA: {args.lora_dir})...")
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if args.device == "cuda" else None,
            trust_remote_code=True,
        )
        if args.lora_dir:
            print(f"Loading LoRA weights from {args.lora_dir}...")
            model = PeftModel.from_pretrained(model, args.lora_dir)
        model.eval()
    except Exception as e:
        print(f"Warning: GPU/transformers dependencies not active or model not loaded ({e}).")
        print("Please ensure torch, transformers, and peft are installed on your GPU machine.")
        return

    test_docs = load_jsonl(args.input_file)
    print(f"Predicting {len(test_docs)} documents...")
    out_docs = []

    for doc in test_docs:
        pmc_id = doc["pmc_id"]
        pmid = doc.get("pmid")
        patients = doc.get("patient", [])

        # Subtask 1: Exact offset matching from dictionary
        extracted = extractor.extract_from_doc(doc)
        pred_entities = []
        candidate_hpos = []
        for ent in extracted:
            pred_entities.append({
                "identifier": ent.identifier,
                "type": ent.type,
                "offset": ent.offset,
                "length": ent.length,
                "text": ent.text,
                "note": None,
            })
            if ent.identifier != "-1":
                candidate_hpos.append(ent.identifier)

        candidate_hpos = sorted(list(set(candidate_hpos)))

        # Extract clinical presentation
        clinical_passages = []
        for sec in doc.get("full_text", []):
            stype = (sec.get("section_type") or "").upper()
            if any(k in stype for k in ["CASE", "PATIENT", "REPORT", "CLINICAL", "PRESENT"]):
                clinical_passages.append(sec.get("text", ""))
        full_clinical_text = "\n\n".join(clinical_passages)
        if not full_clinical_text:
            full_clinical_text = "\n\n".join([s.get("text", "") for s in doc.get("full_text", [])[:5]])

        # Subtask 2: Query LLM for each patient
        pred_associations = []
        for p in patients:
            pid = p["patient_id"]
            mentions = [m.get("text", "") for m in p.get("mention", [])]

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
            import re
            m = re.search(r"\[.*?\]", response, re.DOTALL)
            assigned_hpos = []
            if m:
                try:
                    assigned_hpos = json.loads(m.group(0))
                except Exception:
                    assigned_hpos = []

            # Filter to candidates
            assigned_hpos = [h for h in assigned_hpos if h in candidate_hpos]

            pred_associations.append({
                "patient_id": pid,
                "phenotype": assigned_hpos
            })

        out_docs.append({
            "pmc_id": pmc_id,
            "pmid": pmid,
            "entities": pred_entities,
            "association": pred_associations,
        })

    save_jsonl(out_docs, args.output)
    print(f"Predictions saved to {args.output}")


if __name__ == "__main__":
    main()
