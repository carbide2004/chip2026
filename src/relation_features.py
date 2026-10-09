"""Patient-concept evidence-bag features for relation diagnostics."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple

SECTIONS = ["CASE", "RESULTS", "ABSTRACT", "INTRO", "DISCUSS", "FIG", "METHODS", "CONCL"]


def split_value(identifier: Any, text: str) -> List[str]:
    raw = str(identifier)
    return [text] if raw == "-1" else [part.strip() for part in raw.split(";") if part.strip()]


def passage_index(doc: Dict[str, Any], offset: int) -> int | None:
    for index, passage in enumerate(doc.get("full_text", [])):
        start = passage.get("offset", 0)
        if start <= offset < start + len(passage.get("text", "")):
            return index
    return None


def positive_occurrences(doc: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    output: Dict[str, List[Dict[str, Any]]] = {}
    for entity in doc.get("entities", []):
        if str(entity.get("note") or "").upper() == "NO":
            continue
        index = passage_index(doc, entity["offset"])
        passage = doc.get("full_text", [])[index] if index is not None else {}
        for value in split_value(entity.get("identifier"), entity.get("text", "")):
            output.setdefault(value, []).append(
                {
                    "offset": entity["offset"],
                    "length": entity.get("length", len(entity.get("text", ""))),
                    "text": entity.get("text", ""),
                    "passage_index": index,
                    "section": str(passage.get("section_type") or "").upper(),
                }
            )
    return output


def association_set(row: Dict[str, Any]) -> set[str]:
    values = set()
    for value in row.get("phenotype", []):
        raw = str(value)
        if raw.startswith("HP:") and ";" in raw:
            values.update(part.strip() for part in raw.split(";") if part.strip())
        else:
            values.add(raw)
    return values


def feature_names() -> List[str]:
    return [
        "log_min_distance",
        "log_mean_nearest_distance",
        "same_passage_any",
        "same_passage_occurrence_fraction",
        "patient_is_nearest_any",
        "patient_is_nearest_fraction",
        "occurrence_count_log",
        "patient_anchor_count_log",
        "patient_count_log",
        "single_patient",
        "occurrence_before_anchor_fraction",
        *[f"section_{section.lower()}_fraction" for section in SECTIONS],
        "section_other_fraction",
    ]


def pair_features(
    doc: Dict[str, Any], patient: Dict[str, Any], occurrences: Sequence[Dict[str, Any]]
) -> List[float]:
    patient_id = str(patient["patient_id"])
    anchors = [mention["offset"] for mention in patient.get("mention", [])]
    all_anchors = {
        str(item["patient_id"]): [mention["offset"] for mention in item.get("mention", [])]
        for item in doc.get("patient", [])
    }
    anchor_passages = {passage_index(doc, offset) for offset in anchors}
    distances = [min((abs(item["offset"] - offset) for offset in anchors), default=10**9) for item in occurrences]
    nearest_flags = []
    for item in occurrences:
        by_patient = {
            owner: min((abs(item["offset"] - offset) for offset in offsets), default=10**9)
            for owner, offsets in all_anchors.items()
        }
        nearest_flags.append(by_patient.get(patient_id, 10**9) == min(by_patient.values(), default=10**9))
    count = max(len(occurrences), 1)
    same_passage = [item["passage_index"] in anchor_passages for item in occurrences]
    before = [item["offset"] < min(anchors, default=-1) for item in occurrences]
    section_counts = {section: 0 for section in SECTIONS}
    other = 0
    for item in occurrences:
        section = item["section"]
        matched = next((name for name in SECTIONS if name in section), None)
        if matched:
            section_counts[matched] += 1
        else:
            other += 1
    return [
        math.log1p(min(distances, default=10**9)),
        math.log1p(sum(distances) / count),
        float(any(same_passage)),
        sum(same_passage) / count,
        float(any(nearest_flags)),
        sum(nearest_flags) / count,
        math.log1p(len(occurrences)),
        math.log1p(len(anchors)),
        math.log1p(len(doc.get("patient", []))),
        float(len(doc.get("patient", [])) == 1),
        sum(before) / count,
        *[section_counts[section] / count for section in SECTIONS],
        other / count,
    ]


def build_examples(docs: Sequence[Dict[str, Any]]) -> Tuple[List[List[float]], List[int], List[Tuple[str, str, str]]]:
    features, labels, keys = [], [], []
    for doc in docs:
        occurrences = positive_occurrences(doc)
        gold = {str(row["patient_id"]): association_set(row) for row in doc.get("association", [])}
        for patient in doc.get("patient", []):
            patient_id = str(patient["patient_id"])
            for value, evidence in occurrences.items():
                features.append(pair_features(doc, patient, evidence))
                labels.append(int(value in gold.get(patient_id, set())))
                keys.append((str(doc["pmc_id"]), patient_id, value))
    return features, labels, keys
