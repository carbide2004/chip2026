"""Rule-based and heuristic baseline for CHIP2026 PatientPheX.

Pipeline:
1. Subtask 1: HPO Dictionary maximal matching + basic negation detection.
2. Subtask 2: Heuristic association based on sentence co-occurrence / proximity
   between patient mentions and extracted positive phenotypes.
"""

from pathlib import Path
import sys
from typing import Any, Dict, List, Set

# Ensure repo root in sys.path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from src.data_loader import HPOntology, load_jsonl, save_jsonl
from src.dictionary_extractor import DictionaryExtractor
from src.metrics import compute_all_metrics

# Simple negation triggers for baseline comparison
NEGATION_PATTERNS = [
    r"\bno\b",
    r"\bnot\b",
    r"\bwithout\b",
    r"\bdenies\b",
    r"\bdenied\b",
    r"\babsent\b",
    r"\babsence of\b",
    r"\bnegative for\b",
    r"\bruled out\b",
    r"\bunremarkable\b",
    r"\bnormal\b",
]
import re
NEG_REGEX = re.compile("|".join(NEGATION_PATTERNS), re.IGNORECASE)


class BaselineRulePipeline:
    def __init__(self, obo_path: str):
        self.ontology = HPOntology(obo_path)
        self.extractor = DictionaryExtractor(self.ontology)

    def detect_negation(self, context_sentence: str, entity_text: str) -> bool:
        """Heuristic check whether context sentence indicates negation for entity."""
        # Simple window check: look for negation trigger within 40 chars before entity
        pos = context_sentence.lower().find(entity_text.lower())
        if pos != -1:
            pre_window = context_sentence[max(0, pos - 40):pos]
            post_window = context_sentence[pos + len(entity_text):min(len(context_sentence), pos + len(entity_text) + 30)]
            if NEG_REGEX.search(pre_window) or re.search(r"\b(ruled out|absent|negative)\b", post_window, re.IGNORECASE):
                return True
        return False

    def predict_doc(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        pmc_id = doc["pmc_id"]
        pmid = doc.get("pmid")
        patients = doc.get("patient", [])

        # 1. Subtask 1: Extract entities
        extracted = self.extractor.extract_from_doc(doc)
        pred_entities = []
        positive_entities = []

        for ent in extracted:
            is_neg = self.detect_negation(ent.context_sentence, ent.text)
            note = "NO" if is_neg else None
            ent_dict = {
                "identifier": ent.identifier,
                "type": ent.type,
                "offset": ent.offset,
                "length": ent.length,
                "text": ent.text,
                "note": note,
            }
            pred_entities.append(ent_dict)
            if note is None and ent.identifier != "-1":
                positive_entities.append(ent)

        # 2. Subtask 2: Patient association
        # Heuristic baseline: associate positive phenotypes located in clinical sections
        # or within proximity to patient mentions
        pred_associations = []
        for p in patients:
            pid = p["patient_id"]
            mentions = p.get("mention", [])
            patient_offsets = [m["offset"] for m in mentions]

            associated_hpos: List[str] = []
            seen_hpos: Set[str] = set()

            for ent in positive_entities:
                # Proximity rule: if entity is within 1500 chars of any patient mention
                # or if there is only 1 patient in document, associate it
                is_close = any(abs(ent.offset - p_off) < 1500 for p_off in patient_offsets)
                if len(patients) == 1 or is_close:
                    if ent.identifier not in seen_hpos:
                        seen_hpos.add(ent.identifier)
                        associated_hpos.append(ent.identifier)

            pred_associations.append({
                "patient_id": pid,
                "phenotype": associated_hpos
            })

        return {
            "pmc_id": pmc_id,
            "pmid": pmid,
            "entities": pred_entities,
            "association": pred_associations,
        }

    def predict_all(self, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [self.predict_doc(d) for d in docs]


def main():
    obo_file = str(repo_root / "data" / "raw" / "PatientPheX-V1-A" / "hp.obo")
    train_file = str(repo_root / "data" / "raw" / "PatientPheX-V1-A" / "PatientPheX-train.jsonl")
    a_test_file = str(repo_root / "data" / "raw" / "PatientPheX-V1-A" / "PatientPheX-A.jsonl")

    print(f"Loading ontology from {obo_file}...")
    pipeline = BaselineRulePipeline(obo_file)

    print(f"Loading train docs from {train_file}...")
    train_docs = load_jsonl(train_file)

    print("Running baseline predictions on train set (80 docs)...")
    preds = pipeline.predict_all(train_docs)

    metrics = compute_all_metrics(train_docs, preds)
    print("\n" + "=" * 50)
    print(f"=== Baseline Rule Evaluation (Train Set 80 docs) ===")
    print(f"Overall Benchmark Score : {metrics['overall_score']:.4f}")
    print(f"Subtask 1 Mention F1    : {metrics['subtask1']['f1_men']:.4f} (P: {metrics['subtask1']['p_men']:.4f}, R: {metrics['subtask1']['r_men']:.4f})")
    print(f"Subtask 1 Document F1   : {metrics['subtask1']['f1_doc']:.4f} (P: {metrics['subtask1']['p_doc']:.4f}, R: {metrics['subtask1']['r_doc']:.4f})")
    print(f"Subtask 2 Micro F1      : {metrics['subtask2']['f1_micro']:.4f} (P: {metrics['subtask2']['p_micro']:.4f}, R: {metrics['subtask2']['r_micro']:.4f})")
    print(f"Subtask 2 Macro F1      : {metrics['subtask2']['f1_macro']:.4f} (P: {metrics['subtask2']['p_macro']:.4f}, R: {metrics['subtask2']['r_macro']:.4f})")
    print("=" * 50 + "\n")

    # Generate A-test prediction file
    print(f"Generating A-test predictions from {a_test_file}...")
    a_test_docs = load_jsonl(a_test_file)
    a_preds = pipeline.predict_all(a_test_docs)
    out_a = repo_root / "submissions" / "baseline_rule_A.jsonl"
    save_jsonl(a_preds, str(out_a))
    print(f"A-test submission file successfully saved to: {out_a}")


if __name__ == "__main__":
    main()
