"""Auditable lexical and proximity baselines; no learned components."""

from __future__ import annotations

from collections import defaultdict
import re
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

from .data_loader import HPOntology

_WORD = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*", re.IGNORECASE)
_NEGATION = re.compile(
    r"\b(?:no|not|without|absence of|absent|negative for|denies|denied|free of)\b",
    re.IGNORECASE,
)


def split_identifiers(raw: Any) -> List[str]:
    return [part.strip() for part in str(raw).split(";") if part.strip()]


class LexicalPhenotypeBaseline:
    """Exact HPO name/synonym matcher that preserves source coordinates."""

    def __init__(self, ontology: HPOntology, minimum_length: int = 3):
        self.phrase_to_ids: Dict[str, Set[str]] = {}
        self.name_to_ids: Dict[str, Set[str]] = defaultdict(set)
        for term_id, term in ontology.terms.items():
            if term.is_obsolete or not term.is_sub_abnormality:
                continue
            if term.name:
                self.name_to_ids[self._normal(term.name)].add(term_id)
            for phrase in [term.name, *term.synonyms]:
                normalized = self._normal(phrase)
                if len(normalized) >= minimum_length and not normalized.isdigit():
                    self.phrase_to_ids.setdefault(normalized, set()).add(term_id)

        self.first_token: Dict[str, List[str]] = defaultdict(list)
        for phrase in self.phrase_to_ids:
            match = _WORD.match(phrase)
            if match:
                self.first_token[match.group(0)].append(phrase)
        for phrases in self.first_token.values():
            phrases.sort(key=lambda value: (-len(value), value))

    @staticmethod
    def _normal(text: str) -> str:
        return " ".join(text.lower().split())

    def _select_id(self, phrase: str) -> str:
        ids = self.phrase_to_ids[phrase]
        exact_name_ids = ids & self.name_to_ids.get(phrase, set())
        return sorted(exact_name_ids or ids)[0]

    @staticmethod
    def _is_negated(text: str, start: int) -> bool:
        left = text[max(0, start - 100) : start]
        left = re.split(r"[.;:\n]", left)[-1]
        return bool(_NEGATION.search(left))

    def extract_passage(
        self,
        text: str,
        global_offset: int,
        *,
        overlap_policy: str = "longest",
        detect_negation: bool = False,
    ) -> List[Dict[str, Any]]:
        if overlap_policy not in {"longest", "all"}:
            raise ValueError(f"Unknown overlap policy: {overlap_policy}")
        lower = text.lower()
        occupied: List[Tuple[int, int]] = []
        output: List[Dict[str, Any]] = []
        seen: Set[Tuple[int, int, str]] = set()

        for token in _WORD.finditer(lower):
            start = token.start()
            if overlap_policy == "longest" and any(left <= start < right for left, right in occupied):
                continue
            for phrase in self.first_token.get(token.group(0), []):
                end = start + len(phrase)
                if end > len(text) or lower[start:end] != phrase:
                    continue
                if start > 0 and lower[start - 1].isalnum():
                    continue
                if end < len(text) and lower[end].isalnum():
                    continue
                identifier = self._select_id(phrase)
                key = (start, end, identifier)
                if key in seen:
                    continue
                seen.add(key)
                output.append(
                    {
                        "identifier": identifier,
                        "type": "Phenotype",
                        "offset": global_offset + start,
                        "length": end - start,
                        "text": text[start:end],
                        "note": "NO" if detect_negation and self._is_negated(text, start) else None,
                    }
                )
                if overlap_policy == "longest":
                    occupied.append((start, end))
                    break
        return output

    def extract_document(
        self,
        doc: Dict[str, Any],
        *,
        overlap_policy: str = "longest",
        detect_negation: bool = False,
    ) -> List[Dict[str, Any]]:
        entities = [
            entity
            for passage in doc.get("full_text", [])
            for entity in self.extract_passage(
                passage.get("text", ""),
                passage.get("offset", 0),
                overlap_policy=overlap_policy,
                detect_negation=detect_negation,
            )
        ]
        entities.sort(key=lambda item: (item["offset"], item["length"], item["identifier"]))
        return entities


def _passage_index(doc: Dict[str, Any], offset: int) -> int | None:
    for index, passage in enumerate(doc.get("full_text", [])):
        start = passage.get("offset", 0)
        if start <= offset < start + len(passage.get("text", "")):
            return index
    return None


def associate_by_position(
    doc: Dict[str, Any],
    entities: Sequence[Dict[str, Any]],
    *,
    method: str = "window",
    window: int = 1500,
) -> List[Dict[str, Any]]:
    """Transparent relation baselines used to locate the association bottleneck."""
    patients = doc.get("patient", [])
    anchors = {
        str(patient["patient_id"]): [mention["offset"] for mention in patient.get("mention", [])]
        for patient in patients
    }
    anchor_passages = {
        patient_id: {_passage_index(doc, offset) for offset in offsets}
        for patient_id, offsets in anchors.items()
    }
    values: Dict[str, List[str]] = {str(patient["patient_id"]): [] for patient in patients}
    seen: Dict[str, Set[str]] = {key: set() for key in values}

    for entity in entities:
        if str(entity.get("note") or "").upper() == "NO":
            continue
        entity_offset = entity["offset"]
        entity_passage = _passage_index(doc, entity_offset)
        identifier = str(entity.get("identifier"))
        concept_values = [entity.get("text", "")] if identifier == "-1" else split_identifiers(identifier)
        distances = {
            patient_id: min((abs(entity_offset - offset) for offset in offsets), default=float("inf"))
            for patient_id, offsets in anchors.items()
        }
        selected: Set[str] = set()
        if method == "window":
            selected = {patient_id for patient_id, distance in distances.items() if distance <= window}
        elif method == "same_passage":
            selected = {
                patient_id
                for patient_id, passages in anchor_passages.items()
                if entity_passage is not None and entity_passage in passages
            }
        elif method == "nearest":
            nearest = min(distances.values(), default=float("inf"))
            if nearest <= window:
                selected = {patient_id for patient_id, distance in distances.items() if distance == nearest}
        elif method == "all":
            selected = set(values)
        else:
            raise ValueError(f"Unknown association method: {method}")

        for patient_id in selected:
            for value in concept_values:
                if value and value not in seen[patient_id]:
                    seen[patient_id].add(value)
                    values[patient_id].append(value)

    return [
        {"patient_id": str(patient["patient_id"]), "phenotype": values[str(patient["patient_id"])]}
        for patient in patients
    ]


def build_prediction(
    doc: Dict[str, Any],
    entities: Sequence[Dict[str, Any]],
    *,
    association_method: str,
    association_window: int,
) -> Dict[str, Any]:
    return {
        "pmc_id": doc["pmc_id"],
        "pmid": doc.get("pmid"),
        "entities": list(entities),
        "association": associate_by_position(
            doc, entities, method=association_method, window=association_window
        ),
    }
