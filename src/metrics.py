"""Evaluation metrics for CHIP2026 PatientPheX competition.

Subtask 1: Full-Text Phenotype Concept Recognition
  - Mention-level (P_men, R_men, F1_men): exact match of span (offset, length) and HPO identifier.
    * negated phenotype (note="NO"): not a positive instance. Gold note="NO" does not count toward FN.
      Predicted correctly with note="NO" excluded from eval. Predicted without note="NO" counts as FP.
      Not predicted: ignored.
    * composite phenotype: split by semicolon, each ID evaluated independently.
    * Gold ID = "-1": span match is sufficient. A predicted "-1" is not a wildcard
      for a gold entity that has a valid HPO ID.
  - Document-level (P_doc, R_doc, F1_doc): micro-averaged across documents on unique HPO ID sets.

Subtask 2: Patient-Specific Phenotype Concept Recognition
  - Micro-F1 (P_micro, R_micro, F1_micro): globally accumulated over all (patient, phenotype) pairs.
  - Macro-F1 (P_macro, R_macro, F1_macro): unweighted average across all gold patients.
    If both gold and pred are empty, P=R=F1=1.0.

Overall Score:
  Score = 0.25 * (F1_men + F1_doc) + 0.25 * (F1_micro + F1_macro)
"""

from typing import Any, Dict, List, Set, Tuple


def _split_identifiers(raw_id: Any) -> List[str]:
    """Split composite HPO IDs separated by semicolon into clean distinct IDs."""
    if raw_id is None:
        return []
    s = str(raw_id).strip()
    if not s:
        return []
    return [part.strip() for part in s.split(";") if part.strip()]


def evaluate_subtask1_mention(
    gold_docs: List[Dict[str, Any]], pred_docs: List[Dict[str, Any]]
) -> Dict[str, float]:
    """Calculate mention-level precision, recall, and F1 across all documents."""
    pred_by_doc = {doc["pmc_id"]: doc.get("entities", []) for doc in pred_docs}

    tp_total = 0
    fp_total = 0
    fn_total = 0

    for gold_doc in gold_docs:
        pmc_id = gold_doc["pmc_id"]
        gold_entities = gold_doc.get("entities", [])
        pred_entities = pred_by_doc.get(pmc_id, [])

        # Parse gold mentions:
        # Tuple: (offset, length, id_token)
        # Note handling: gold note == "NO" is excluded from positives (does NOT count toward FN)
        gold_positives: Set[Tuple[int, int, str]] = set()
        gold_negations: Set[Tuple[int, int]] = set()

        for ent in gold_entities:
            offset = ent["offset"]
            length = ent["length"]
            note = (ent.get("note") or "").strip().upper()
            if note == "NO":
                gold_negations.add((offset, length))
                continue
            for id_tok in _split_identifiers(ent.get("identifier")):
                gold_positives.add((offset, length, id_tok))

        # Parse predicted mentions
        pred_positives: Set[Tuple[int, int, str]] = set()
        for ent in pred_entities:
            offset = ent["offset"]
            length = ent["length"]
            note = (ent.get("note") or "").strip().upper()
            # If predicted with note="NO":
            # If gold is indeed negation -> properly excluded (no penalty)
            # If gold is not negation -> not counted as positive, so does not generate positive TP/FP
            if note == "NO":
                continue

            # Model predicted regular phenotype without note="NO"
            for id_tok in _split_identifiers(ent.get("identifier")):
                pred_positives.add((offset, length, id_tok))

        # Matching is one-to-one at the split identifier-unit level. A gold
        # no-ID entity is span-only, but predicting -1 must not turn a valid
        # gold HPO identifier into a true positive.
        matched_gold: Set[Tuple[int, int, str]] = set()
        matched_pred: Set[Tuple[int, int, str]] = set()

        for p in sorted(pred_positives):
            p_off, p_len, p_id = p
            for g in sorted(gold_positives):
                if g in matched_gold:
                    continue
                g_off, g_len, g_id = g
                if p_off == g_off and p_len == g_len:
                    if g_id == "-1" or p_id == g_id:
                        matched_gold.add(g)
                        matched_pred.add(p)
                        break

        tp = len(matched_pred)
        fp = len(pred_positives) - tp
        fn = len(gold_positives) - len(matched_gold)

        tp_total += tp
        fp_total += fp
        fn_total += fn

    p = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    r = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    return {
        "p_men": p,
        "r_men": r,
        "f1_men": f1,
        "tp_men": tp_total,
        "fp_men": fp_total,
        "fn_men": fn_total,
    }


def evaluate_subtask1_doc(
    gold_docs: List[Dict[str, Any]], pred_docs: List[Dict[str, Any]]
) -> Dict[str, float]:
    """Calculate document-level precision, recall, and F1 across all documents."""
    pred_by_doc = {doc["pmc_id"]: doc.get("entities", []) for doc in pred_docs}

    tp_total = 0
    fp_total = 0
    fn_total = 0

    for gold_doc in gold_docs:
        pmc_id = gold_doc["pmc_id"]
        gold_entities = gold_doc.get("entities", [])
        pred_entities = pred_by_doc.get(pmc_id, [])

        g_d: Set[str] = set()
        for ent in gold_entities:
            note = (ent.get("note") or "").strip().upper()
            if note == "NO":
                continue
            for id_tok in _split_identifiers(ent.get("identifier")):
                if id_tok != "-1":
                    g_d.add(id_tok)

        p_d: Set[str] = set()
        for ent in pred_entities:
            note = (ent.get("note") or "").strip().upper()
            if note == "NO":
                continue
            for id_tok in _split_identifiers(ent.get("identifier")):
                if id_tok != "-1":
                    p_d.add(id_tok)

        tp = len(g_d & p_d)
        fp = len(p_d - g_d)
        fn = len(g_d - p_d)

        tp_total += tp
        fp_total += fp
        fn_total += fn

    p = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    r = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    return {
        "p_doc": p,
        "r_doc": r,
        "f1_doc": f1,
        "tp_doc": tp_total,
        "fp_doc": fp_total,
        "fn_doc": fn_total,
    }


def evaluate_subtask2(
    gold_docs: List[Dict[str, Any]], pred_docs: List[Dict[str, Any]]
) -> Dict[str, float]:
    """Calculate Subtask 2 Micro-F1 and Macro-F1 across all patients."""
    pred_by_doc: Dict[str, Dict[str, Set[str]]] = {}
    for doc in pred_docs:
        pmc_id = doc["pmc_id"]
        patient_map: Dict[str, Set[str]] = {}
        for assoc in doc.get("association", []):
            pid = assoc.get("patient_id")
            pheno_list = assoc.get("phenotype", [])
            tokens: Set[str] = set()
            for item in pheno_list:
                for t in _split_identifiers(item):
                    tokens.add(t)
            patient_map[pid] = tokens
        pred_by_doc[pmc_id] = patient_map

    total_tp = 0
    total_fp = 0
    total_fn = 0

    patient_p_list: List[float] = []
    patient_r_list: List[float] = []
    patient_f1_list: List[float] = []

    for gold_doc in gold_docs:
        pmc_id = gold_doc["pmc_id"]
        doc_preds = pred_by_doc.get(pmc_id, {})

        for assoc in gold_doc.get("association", []):
            pid = assoc.get("patient_id")
            gold_phenos: Set[str] = set()
            for item in assoc.get("phenotype", []):
                for t in _split_identifiers(item):
                    gold_phenos.add(t)

            pred_phenos = doc_preds.get(pid, set())

            tp = len(gold_phenos & pred_phenos)
            fp = len(pred_phenos - gold_phenos)
            fn = len(gold_phenos - pred_phenos)

            total_tp += tp
            total_fp += fp
            total_fn += fn

            # Patient-level Macro metrics
            if not gold_phenos and not pred_phenos:
                pi, ri, f1i = 1.0, 1.0, 1.0
            else:
                pi = tp / len(pred_phenos) if pred_phenos else 0.0
                ri = tp / len(gold_phenos) if gold_phenos else 0.0
                f1i = (2 * pi * ri / (pi + ri)) if (pi + ri) > 0 else 0.0

            patient_p_list.append(pi)
            patient_r_list.append(ri)
            patient_f1_list.append(f1i)

    # Micro-metrics
    p_micro = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    r_micro = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1_micro = (
        2 * p_micro * r_micro / (p_micro + r_micro)
        if (p_micro + r_micro) > 0
        else 0.0
    )

    # Macro-metrics
    k = len(patient_f1_list)
    p_macro = sum(patient_p_list) / k if k > 0 else 0.0
    r_macro = sum(patient_r_list) / k if k > 0 else 0.0
    f1_macro = sum(patient_f1_list) / k if k > 0 else 0.0

    return {
        "p_micro": p_micro,
        "r_micro": r_micro,
        "f1_micro": f1_micro,
        "p_macro": p_macro,
        "r_macro": r_macro,
        "f1_macro": f1_macro,
        "gold_patient_count": k,
    }


def compute_all_metrics(
    gold_docs: List[Dict[str, Any]], pred_docs: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Compute official benchmark overall score and full component breakdown."""
    st1_men = evaluate_subtask1_mention(gold_docs, pred_docs)
    st1_doc = evaluate_subtask1_doc(gold_docs, pred_docs)
    st2 = evaluate_subtask2(gold_docs, pred_docs)

    overall_score = 0.25 * (st1_men["f1_men"] + st1_doc["f1_doc"]) + 0.25 * (
        st2["f1_micro"] + st2["f1_macro"]
    )

    return {
        "overall_score": overall_score,
        "subtask1": {**st1_men, **st1_doc},
        "subtask2": st2,
    }
