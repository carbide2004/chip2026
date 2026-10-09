"""Submission validation independent of model inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any, Dict, List, Sequence, Set, Tuple

_HPO_OR_NO_ID = re.compile(r"^(?:HP:\d{7})(?:;HP:\d{7})*$|^-1$")


@dataclass(frozen=True)
class ValidationIssue:
    severity: str
    code: str
    pmc_id: str = ""
    detail: str = ""

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)


def _source_text(doc: Dict[str, Any], offset: int, length: int) -> str | None:
    for passage in doc.get("full_text", []):
        start = passage.get("offset")
        text = passage.get("text")
        if not isinstance(start, int) or not isinstance(text, str):
            continue
        local = offset - start
        if 0 <= local and local + length <= len(text):
            return text[local : local + length]
    return None


def validate_submission(
    input_docs: Sequence[Dict[str, Any]], prediction_docs: Sequence[Dict[str, Any]]
) -> List[ValidationIssue]:
    """Check structure, coverage and source-coordinate integrity."""
    issues: List[ValidationIssue] = []
    source_by_id = {str(doc.get("pmc_id")): doc for doc in input_docs}
    prediction_ids = [str(doc.get("pmc_id")) for doc in prediction_docs]
    if len(prediction_ids) != len(set(prediction_ids)):
        issues.append(ValidationIssue("error", "duplicate_prediction_document"))

    missing = set(source_by_id) - set(prediction_ids)
    extra = set(prediction_ids) - set(source_by_id)
    for pmc_id in sorted(missing):
        issues.append(ValidationIssue("error", "missing_document", pmc_id))
    for pmc_id in sorted(extra):
        issues.append(ValidationIssue("error", "extra_document", pmc_id))

    for prediction in prediction_docs:
        pmc_id = str(prediction.get("pmc_id"))
        source = source_by_id.get(pmc_id)
        if source is None:
            continue
        if prediction.get("pmid") != source.get("pmid"):
            issues.append(ValidationIssue("error", "pmid_mismatch", pmc_id))

        expected_patients = {str(p.get("patient_id")) for p in source.get("patient", [])}
        rows = prediction.get("association")
        if not isinstance(rows, list):
            issues.append(ValidationIssue("error", "association_not_list", pmc_id))
            rows = []
        actual_patients = [str(row.get("patient_id")) for row in rows if isinstance(row, dict)]
        if len(actual_patients) != len(set(actual_patients)):
            issues.append(ValidationIssue("error", "duplicate_association_patient", pmc_id))
        for patient_id in sorted(expected_patients - set(actual_patients)):
            issues.append(ValidationIssue("error", "missing_patient", pmc_id, patient_id))
        for patient_id in sorted(set(actual_patients) - expected_patients):
            issues.append(ValidationIssue("error", "extra_patient", pmc_id, patient_id))
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("phenotype"), list):
                issues.append(ValidationIssue("error", "invalid_association_row", pmc_id))
                continue
            values = row["phenotype"]
            if len(values) != len(set(values)):
                issues.append(ValidationIssue("warning", "duplicate_association_value", pmc_id, str(row.get("patient_id"))))
            if any(not isinstance(value, str) or not value for value in values):
                issues.append(ValidationIssue("error", "invalid_association_value", pmc_id, str(row.get("patient_id"))))

        entities = prediction.get("entities")
        if not isinstance(entities, list):
            issues.append(ValidationIssue("error", "entities_not_list", pmc_id))
            continue
        seen: Set[Tuple[int, int, str, str]] = set()
        for index, entity in enumerate(entities):
            if not isinstance(entity, dict):
                issues.append(ValidationIssue("error", "entity_not_object", pmc_id, str(index)))
                continue
            identifier = str(entity.get("identifier", ""))
            offset, length, text = entity.get("offset"), entity.get("length"), entity.get("text")
            note = entity.get("note")
            if not _HPO_OR_NO_ID.fullmatch(identifier):
                issues.append(ValidationIssue("error", "invalid_entity_identifier", pmc_id, f"{index}:{identifier}"))
            if entity.get("type") != "Phenotype":
                issues.append(ValidationIssue("error", "invalid_entity_type", pmc_id, str(index)))
            if note not in (None, "NO"):
                issues.append(ValidationIssue("error", "invalid_entity_note", pmc_id, str(index)))
            if not isinstance(offset, int) or not isinstance(length, int) or length <= 0 or not isinstance(text, str):
                issues.append(ValidationIssue("error", "invalid_entity_span", pmc_id, str(index)))
                continue
            source_text = _source_text(source, offset, length)
            if source_text != text:
                issues.append(ValidationIssue("error", "entity_text_offset_mismatch", pmc_id, str(index)))
            key = (offset, length, identifier, str(note))
            if key in seen:
                issues.append(ValidationIssue("warning", "duplicate_entity", pmc_id, str(index)))
            seen.add(key)
    return issues


def raise_for_errors(issues: Sequence[ValidationIssue]) -> None:
    errors = [issue for issue in issues if issue.severity == "error"]
    if errors:
        preview = "; ".join(f"{x.code}:{x.pmc_id}:{x.detail}" for x in errors[:10])
        raise ValueError(f"Submission validation failed with {len(errors)} error(s): {preview}")
