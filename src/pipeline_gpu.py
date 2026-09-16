"""GPU-enhanced Pipeline for CHIP2026 PatientPheX using SapBERT & Embedding Matching.
"""

from pathlib import Path
import os
import sys
import re
from typing import Any, Dict, List, Set, Tuple
import numpy as np

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from src.data_loader import HPOntology, load_jsonl, save_jsonl
from src.dictionary_extractor import DictionaryExtractor, ExtractedEntity
from src.metrics import compute_all_metrics
from src.sapbert_matcher import SapBERTRetriever

# Common biomedical suffixes/keywords that suggest phenotypic abnormality
BIO_PATTERNS = re.compile(
    r"\b[a-zA-Z0-9\-]+(?:itis|osis|pathy|trophy|plasia|emia|uria|oma|stasis|"
    r"malformation|defect|deficiency|delay|delays|disability|disabilities|"
    r"abnormality|abnormalities|impairment|syndrome|disease|failure|pain|loss|"
    r"decline|atrophy|hypertrophy|spasm|weakness|convulsion|seizure|seizures|"
    r"fever|edema|swelling|lesion|lesions|infection|infections|dysfunction)\b",
    re.IGNORECASE,
)

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "in", "to", "with", "by", "for", "on",
    "at", "from", "as", "is", "was", "were", "are", "be", "been", "that", "this",
    "these", "those", "patient", "patients", "case", "cases", "report", "study",
}


class GPUEnhancedPipeline:
    def __init__(self, obo_path: str, sim_threshold: float = 0.90):
        self.sim_threshold = sim_threshold
        print(f"Loading HPO ontology from {obo_path}...")
        self.ontology = HPOntology(obo_path)
        self.dict_extractor = DictionaryExtractor(self.ontology)
        print("Loading SapBERT GPU Retriever...")
        self.retriever = SapBERTRetriever(obo_path=obo_path)
        if not self.retriever.load_index():
            self.retriever.build_index(batch_size=512)

    def detect_negation(self, context_sentence: str, entity_text: str) -> bool:
        """Enhanced rule-based negation and normality detection."""
        pos = context_sentence.lower().find(entity_text.lower())
        if pos != -1:
            pre_window = context_sentence[max(0, pos - 60):pos]
            post_window = context_sentence[pos + len(entity_text):min(len(context_sentence), pos + len(entity_text) + 40)]
            neg_pat = (
                r"\b(no|not|none|without|denies|denied|absent|absence of|negative for|free of|"
                r"unremarkable|normal|within normal limits|ruled out|unaffected|healthy|"
                r"no history of|no signs of|no evidence of|failed to reveal|neither|nor)\b"
            )
            post_pat = (
                r"\b(ruled out|absent|negative|unremarkable|normal|were excluded|was excluded|"
                r"not observed|not detected|were unremarkable|was unremarkable|within normal limits)\b"
            )
            if re.search(neg_pat, pre_window, re.IGNORECASE) or re.search(post_pat, post_window, re.IGNORECASE):
                return True
        return False

    def extract_candidates_from_text(
        self, text: str, occupied_spans: List[Tuple[int, int]]
    ) -> List[Tuple[int, int, str]]:
        """Extract candidate phrases likely to be biomedical abnormalities."""
        candidates = []
        n = len(text)
        words = list(re.finditer(r"\b[a-zA-Z0-9\-]+\b", text))

        for i, w_match in enumerate(words):
            start = w_match.start()
            if any(s <= start < e for s, e in occupied_spans):
                continue
            
            # Lookahead up to 4 words
            for j in range(i, min(i + 4, len(words))):
                end = words[j].end()
                if any(s < end <= e for s, e in occupied_spans):
                    break
                phrase = text[start:end].strip()
                tokens = phrase.lower().split()
                if not tokens or tokens[0] in STOPWORDS or tokens[-1] in STOPWORDS:
                    continue
                if len(phrase) < 4:
                    continue
                # If phrase contains biomedical key indicator or plural term
                if BIO_PATTERNS.search(phrase) or phrase.endswith(("ies", "es", "s")):
                    candidates.append((start, end, phrase))
        return candidates

    def predict_doc(self, doc: Dict[str, Any], use_sapbert: bool = True) -> Dict[str, Any]:
        pmc_id = doc["pmc_id"]
        pmid = doc.get("pmid")
        patients = doc.get("patient", [])

        # 1. First pass: exact dictionary matching
        dict_entities = self.dict_extractor.extract_from_doc(doc)
        
        all_pred_entities: List[Dict[str, Any]] = []
        positive_entities: List[Dict[str, Any]] = []

        # Track occupied spans per section
        doc_spans: Dict[int, List[Tuple[int, int]]] = {}
        for ent in dict_entities:
            doc_spans.setdefault(ent.offset, []).append((ent.offset, ent.offset + ent.length))

        # Add exact entities
        for ent in dict_entities:
            is_neg = self.detect_negation(ent.context_sentence, ent.text)
            note = "NO" if is_neg else None
            item = {
                "identifier": ent.identifier,
                "type": ent.type,
                "offset": ent.offset,
                "length": ent.length,
                "text": ent.text,
                "note": note,
            }
            all_pred_entities.append(item)
            if note is None and ent.identifier != "-1":
                positive_entities.append(item)

        # 2. Second pass (GPU SapBERT): Soft matching for missed variants
        if use_sapbert:
            all_candidate_queries = []
            all_candidate_meta = []

            for sec in doc.get("full_text", []):
                sec_text = sec.get("text", "")
                sec_offset = sec.get("offset", 0)
                sec_type = (sec.get("section_type") or "").upper()
                if any(skip in sec_type for skip in ["REF", "ACK", "FOOTNOTE", "SUPPL", "AUTHOR"]):
                    continue

                occupied = [
                    (e["offset"] - sec_offset, e["offset"] + e["length"] - sec_offset)
                    for e in all_pred_entities
                    if sec_offset <= e["offset"] < sec_offset + len(sec_text)
                ]

                cand_spans = self.extract_candidates_from_text(sec_text, occupied)
                for start, end, phrase in cand_spans:
                    all_candidate_queries.append(phrase)
                    all_candidate_meta.append((sec_offset + start, end - start, phrase, sec_text, start, end))

            # Batch retrieve with SapBERT
            if all_candidate_queries:
                batch_results = self.retriever.search_candidates_batch(
                    all_candidate_queries, top_k=1, batch_size=256
                )
                for (global_start, length, phrase, sec_text, start, end), top_matches in zip(
                    all_candidate_meta, batch_results
                ):
                    if not top_matches:
                        continue
                    hpo_id, hpo_name, score = top_matches[0]
                    if score >= self.sim_threshold:
                        # Negation check
                        ctx_start = max(0, start - 60)
                        ctx_end = min(len(sec_text), end + 60)
                        context = sec_text[ctx_start:ctx_end].replace("\n", " ").strip()
                        is_neg = self.detect_negation(context, phrase)
                        note = "NO" if is_neg else None

                        item = {
                            "identifier": hpo_id,
                            "type": "Phenotype",
                            "offset": global_start,
                            "length": length,
                            "text": phrase,
                            "note": note,
                        }
                        all_pred_entities.append(item)
                        if note is None and hpo_id != "-1":
                            positive_entities.append(item)

        # De-duplicate spans (keep longest/earliest)
        seen_spans = set()
        dedup_entities = []
        all_pred_entities.sort(key=lambda x: (x["offset"], -x["length"]))
        for e in all_pred_entities:
            span_key = (e["offset"], e["length"])
            if span_key not in seen_spans:
                seen_spans.add(span_key)
                dedup_entities.append(e)

        # 3. Subtask 2: Association
        pred_associations = []
        for p in patients:
            pid = p["patient_id"]
            mentions = p.get("mention", [])
            patient_offsets = [m["offset"] for m in mentions]

            associated_hpos = []
            seen_hpos = set()
            for ent in positive_entities:
                is_close = any(abs(ent["offset"] - p_off) < 1500 for p_off in patient_offsets)
                if len(patients) == 1 or is_close:
                    h_id = ent["identifier"]
                    if h_id not in seen_hpos:
                        seen_hpos.add(h_id)
                        associated_hpos.append(h_id)

            pred_associations.append({"patient_id": pid, "phenotype": associated_hpos})

        return {
            "pmc_id": pmc_id,
            "pmid": pmid,
            "entities": dedup_entities,
            "association": pred_associations,
        }


def format_metrics(res: Dict[str, Any]) -> Dict[str, float]:
    st1 = res.get("subtask1", {})
    st2 = res.get("subtask2", {})
    return {
        "overall_score": res.get("overall_score", 0.0),
        "s1_mention_f1": st1.get("f1_men", 0.0),
        "s1_mention_p": st1.get("p_men", 0.0),
        "s1_mention_r": st1.get("r_men", 0.0),
        "s1_doc_f1": st1.get("f1_doc", 0.0),
        "s1_doc_p": st1.get("p_doc", 0.0),
        "s1_doc_r": st1.get("r_doc", 0.0),
        "s2_micro_f1": st2.get("f1_micro", 0.0),
        "s2_micro_p": st2.get("p_micro", 0.0),
        "s2_micro_r": st2.get("r_micro", 0.0),
        "s2_macro_f1": st2.get("f1_macro", 0.0),
        "s2_macro_p": st2.get("p_macro", 0.0),
        "s2_macro_r": st2.get("r_macro", 0.0),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_docs", type=int, default=80, help="Number of train docs to evaluate (max 80)")
    parser.add_argument("--threshold", type=float, default=0.92, help="SapBERT cosine similarity threshold")
    parser.add_argument("--save_submission", action="store_true", help="Generate submission for test A set")
    args = parser.parse_args()

    train_path = repo_root / "data" / "raw" / "PatientPheX-V1-A" / "PatientPheX-train.jsonl"
    obo_path = repo_root / "data" / "raw" / "PatientPheX-V1-A" / "hp.obo"
    
    docs = load_jsonl(train_path)
    eval_docs = docs[: args.num_docs]
    gold_docs = eval_docs

    print(f"\n--- Running GPU Pipeline on {len(eval_docs)} documents ---")
    pipeline = GPUEnhancedPipeline(str(obo_path), sim_threshold=args.threshold)

    # 1. Without SapBERT (Dictionary only baseline)
    print("Evaluating Rule Baseline...")
    pred_base = [pipeline.predict_doc(d, use_sapbert=False) for d in eval_docs]
    m_base = format_metrics(compute_all_metrics(gold_docs, pred_base))

    # 2. With SapBERT (GPU soft matching)
    print("Evaluating GPU SapBERT Enhanced Pipeline...")
    pred_gpu = [pipeline.predict_doc(d, use_sapbert=True) for d in eval_docs]
    m_gpu = format_metrics(compute_all_metrics(gold_docs, pred_gpu))

    print("\n" + "=" * 67)
    print("=== Comparison: Rule Baseline vs GPU SapBERT Enhanced ===")
    print(f"{'Metric':<20} | {'Baseline':<12} | {'GPU SapBERT':<12} | {'Diff':<8}")
    print("-" * 67)
    for k in [
        "overall_score",
        "s1_mention_f1",
        "s1_mention_p",
        "s1_mention_r",
        "s1_doc_f1",
        "s1_doc_p",
        "s1_doc_r",
        "s2_micro_f1",
        "s2_micro_p",
        "s2_micro_r",
        "s2_macro_f1",
        "s2_macro_p",
        "s2_macro_r",
    ]:
        v1 = m_base.get(k, 0.0)
        v2 = m_gpu.get(k, 0.0)
        diff = v2 - v1
        print(f"{k:<20} | {v1:<12.4f} | {v2:<12.4f} | {diff:+.4f}")
    print("=" * 67)

    if args.save_submission:
        test_path = repo_root / "data" / "raw" / "PatientPheX-V1-A" / "PatientPheX-A.jsonl"
        out_path = repo_root / "submissions" / "sapbert_rule_A.jsonl"
        print(f"\nGenerating A-test predictions from {test_path}...")
        test_docs = load_jsonl(test_path)
        pred_test = [pipeline.predict_doc(d, use_sapbert=True) for d in test_docs]
        save_jsonl(pred_test, out_path)
        print(f"A-test submission file saved to: {out_path}")


if __name__ == "__main__":
    main()
