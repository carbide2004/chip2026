"""Construct SFT instruction-tuning datasets for 8B models (e.g. Qwen2.5-7B/8B).

Generates two datasets in standard Alpaca format:
1. data/processed/sft_negation.json: Sentence-level phenotype negation classification (null vs NO).
2. data/processed/sft_association.json: Patient clinical history context -> Associated HPO IDs.
"""

from pathlib import Path
import json
import sys
from typing import Any, Dict, List, Set

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from src.data_loader import load_jsonl


def build_negation_sft(train_docs: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Build sentence-level negation classification examples."""
    examples = []
    for doc in train_docs:
        full_text_list = doc.get("full_text", [])
        # Map character offsets to sentence/passage text
        for ent in doc.get("entities", []):
            ent_off = ent["offset"]
            ent_len = ent["length"]
            ent_text = ent["text"]
            gold_note = ent.get("note")
            label = "NO" if gold_note == "NO" else "AFFIRMATIVE"

            # Find containing section
            sec_text = ""
            for sec in full_text_list:
                s_off = sec.get("offset", 0)
                s_txt = sec.get("text", "")
                if s_off <= ent_off < s_off + len(s_txt):
                    rel_off = ent_off - s_off
                    start = max(0, rel_off - 120)
                    end = min(len(s_txt), rel_off + ent_len + 120)
                    sec_text = s_txt[start:end].replace("\n", " ").strip()
                    break

            if sec_text:
                instruction = (
                    "Determine whether the target clinical phenotype in the following medical passage "
                    "is negated (absent, ruled out, or normal) or confirmed/present (affirmative). "
                    "Output either 'NO' (for negated) or 'AFFIRMATIVE' (for present)."
                )
                input_str = f"Passage: {sec_text}\nTarget Phenotype: {ent_text}"
                examples.append(
                    {
                        "instruction": instruction,
                        "input": input_str,
                        "output": label,
                    }
                )

    return examples


def build_association_sft(train_docs: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Build patient-phenotype relation extraction examples."""
    examples = []

    for doc in train_docs:
        patients = doc.get("patient", [])
        gold_assocs = {
            a["patient_id"]: a.get("phenotype", [])
            for a in doc.get("association", [])
        }

        # Extract case description passages
        clinical_passages = []
        for sec in doc.get("full_text", []):
            stype = (sec.get("section_type") or "").upper()
            if any(k in stype for k in ["CASE", "PATIENT", "REPORT", "CLINICAL", "PRESENT"]):
                clinical_passages.append(sec.get("text", ""))

        full_clinical_text = "\n\n".join(clinical_passages)
        if not full_clinical_text:
            # Fallback to abstract + front
            full_clinical_text = "\n\n".join([s.get("text", "") for s in doc.get("full_text", [])[:5]])

        # Collect candidate phenotypes in document
        candidate_hpos = sorted(
            list(
                set(
                    e["identifier"]
                    for e in doc.get("entities", [])
                    if e.get("note") != "NO" and e.get("identifier") != "-1"
                )
            )
        )

        for p in patients:
            pid = p["patient_id"]
            mentions = [m.get("text", "") for m in p.get("mention", [])]
            true_hpos = gold_assocs.get(pid, [])

            instruction = (
                "You are an expert clinical geneticist. Below is the clinical presentation from a biomedical article "
                f"and information about patient '{pid}' (mentioned in text as: {', '.join(mentions)}). "
                "From the provided list of candidate HPO phenotype IDs present in the article, determine which "
                "phenotypes are specifically confirmed and belong to this patient. "
                "Exclude phenotypes belonging to relatives, negated symptoms, or healthy traits. "
                "Output your answer as a JSON array of confirmed HPO IDs."
            )

            input_str = (
                f"Candidate HPO IDs: {json.dumps(candidate_hpos)}\n\n"
                f"Clinical Text:\n{full_clinical_text[:4000]}"  # Bounded to avoid exceeding prompt limit
            )

            output_str = json.dumps(true_hpos, ensure_ascii=False)

            examples.append(
                {
                    "instruction": instruction,
                    "input": input_str,
                    "output": output_str,
                }
            )

    return examples


def main():
    train_file = str(repo_root / "data" / "raw" / "PatientPheX-V1-A" / "PatientPheX-train.jsonl")
    print(f"Loading training data from {train_file}...")
    train_docs = load_jsonl(train_file)

    out_dir = repo_root / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Building SFT negation dataset...")
    neg_examples = build_negation_sft(train_docs)
    neg_file = out_dir / "sft_negation.json"
    with open(neg_file, "w", encoding="utf-8") as f:
        json.dump(neg_examples, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(neg_examples)} negation SFT samples to {neg_file}")

    print("Building SFT patient association dataset...")
    assoc_examples = build_association_sft(train_docs)
    assoc_file = out_dir / "sft_association.json"
    with open(assoc_file, "w", encoding="utf-8") as f:
        json.dump(assoc_examples, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(assoc_examples)} patient association SFT samples to {assoc_file}")


if __name__ == "__main__":
    main()
