"""Reproducible data audit and document-level split utilities for PatientPheX."""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import random
import re
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

from .data_loader import HPOntology, load_jsonl

_HPO_ID = re.compile(r"^HP:\d{7}$")


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_identifier(raw: Any) -> List[str]:
    return [part.strip() for part in str(raw).split(";") if part.strip()]


def _span_matches_passage(doc: Dict[str, Any], item: Dict[str, Any]) -> bool:
    offset, length, text = item.get("offset"), item.get("length"), item.get("text")
    if not isinstance(offset, int) or not isinstance(length, int) or not isinstance(text, str):
        return False
    for passage in doc.get("full_text", []):
        start = passage.get("offset")
        raw = passage.get("text")
        if not isinstance(start, int) or not isinstance(raw, str):
            continue
        local = offset - start
        if 0 <= local and local + length <= len(raw) and raw[local : local + length] == text:
            return True
    return False


def audit_dataset(docs: Sequence[Dict[str, Any]], ontology: HPOntology) -> Dict[str, Any]:
    """Return machine-readable counts and issue locations without changing labels."""
    issues: List[Dict[str, Any]] = []
    counters: Counter[str] = Counter()
    seen_documents: Set[str] = set()

    def add(code: str, doc: Dict[str, Any], **details: Any) -> None:
        issues.append({"code": code, "pmc_id": doc.get("pmc_id"), **details})
        counters[code] += 1

    for doc in docs:
        pmc_id = str(doc.get("pmc_id", ""))
        if not pmc_id or pmc_id in seen_documents:
            add("duplicate_or_empty_pmc_id", doc)
        seen_documents.add(pmc_id)

        patient_ids = [p.get("patient_id") for p in doc.get("patient", [])]
        association_ids = [a.get("patient_id") for a in doc.get("association", [])]
        if len(patient_ids) != len(set(patient_ids)):
            add("duplicate_patient_id", doc)
        if set(patient_ids) != set(association_ids):
            add("patient_association_id_mismatch", doc)

        for patient_index, patient in enumerate(doc.get("patient", [])):
            if not patient.get("mention"):
                add("patient_without_supplied_mention", doc, patient_index=patient_index)
            for mention_index, mention in enumerate(patient.get("mention", [])):
                if not _span_matches_passage(doc, mention):
                    add(
                        "patient_mention_span_mismatch",
                        doc,
                        patient_index=patient_index,
                        mention_index=mention_index,
                    )

        for entity_index, entity in enumerate(doc.get("entities", [])):
            counters["entity_count"] += 1
            if not _span_matches_passage(doc, entity):
                add("entity_span_mismatch", doc, entity_index=entity_index)
            if str(entity.get("note") or "").upper() == "NO":
                counters["negated_entity_count"] += 1
            raw_identifier = entity.get("identifier")
            if ";" in str(raw_identifier):
                counters["composite_entity_count"] += 1
            for identifier in split_identifier(raw_identifier):
                if identifier == "-1":
                    counters["no_id_entity_count"] += 1
                    continue
                if not _HPO_ID.fullmatch(identifier):
                    add("malformed_or_non_hpo_identifier", doc, entity_index=entity_index, identifier=identifier)
                    continue
                if identifier in ontology.alt_id_to_id:
                    add(
                        "alternative_hpo_identifier",
                        doc,
                        entity_index=entity_index,
                        identifier=identifier,
                        canonical=ontology.alt_id_to_id[identifier],
                    )
                    continue
                term = ontology.terms.get(identifier)
                if term is None:
                    add("unknown_hpo_identifier", doc, entity_index=entity_index, identifier=identifier)
                elif term.is_obsolete:
                    add(
                        "obsolete_hpo_identifier",
                        doc,
                        entity_index=entity_index,
                        identifier=identifier,
                        replaced_by=term.replaced_by,
                        consider=term.consider,
                    )
                elif not term.is_sub_abnormality:
                    add("hpo_outside_phenotypic_abnormality", doc, entity_index=entity_index, identifier=identifier)

        entity_values: Set[str] = set()
        for entity in doc.get("entities", []):
            if str(entity.get("note") or "").upper() == "NO":
                continue
            for identifier in split_identifier(entity.get("identifier")):
                entity_values.add(entity.get("text", "") if identifier == "-1" else identifier)

        for association_index, association in enumerate(doc.get("association", [])):
            values = association.get("phenotype", [])
            counters["association_row_count"] += 1
            counters["association_raw_value_count"] += len(values)
            duplicates = len(values) - len(set(values))
            if duplicates:
                add(
                    "duplicate_association_value",
                    doc,
                    association_index=association_index,
                    duplicate_count=duplicates,
                )
            for value in values:
                tokens = split_identifier(value) if all(_HPO_ID.fullmatch(x) for x in split_identifier(value)) else [value]
                for token in tokens:
                    if token not in entity_values:
                        add(
                            "association_value_without_positive_entity",
                            doc,
                            association_index=association_index,
                            value=token,
                        )

    return {
        "summary": {
            "document_count": len(docs),
            "patient_count": sum(len(d.get("patient", [])) for d in docs),
            **dict(sorted(counters.items())),
            "issue_count": len(issues),
        },
        "issues": issues,
    }


def document_features(doc: Dict[str, Any]) -> Dict[str, int]:
    entities = doc.get("entities", [])
    return {
        "documents": 1,
        "patients": len(doc.get("patient", [])),
        "multi_patient_documents": int(len(doc.get("patient", [])) > 1),
        "characters": sum(len(p.get("text", "")) for p in doc.get("full_text", [])),
        "entities": len(entities),
        "negated": sum(str(e.get("note") or "").upper() == "NO" for e in entities),
        "composite": sum(";" in str(e.get("identifier", "")) for e in entities),
        "no_id": sum(str(e.get("identifier")) == "-1" for e in entities),
    }


def _word_shingles(doc: Dict[str, Any], width: int = 5) -> Set[str]:
    text = " ".join(p.get("text", "") for p in doc.get("full_text", []))
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {" ".join(words[i : i + width]) for i in range(max(0, len(words) - width + 1))}


def find_near_duplicate_groups(
    docs: Sequence[Dict[str, Any]], threshold: float = 0.8
) -> List[List[str]]:
    """Find small near-duplicate groups by word 5-gram Jaccard similarity."""
    shingles = [_word_shingles(doc) for doc in docs]
    parent = list(range(len(docs)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        left, right = find(i), find(j)
        if left != right:
            parent[right] = left

    for i, left in enumerate(shingles):
        for j in range(i + 1, len(shingles)):
            right = shingles[j]
            if not left or not right:
                continue
            size_ratio = min(len(left), len(right)) / max(len(left), len(right))
            if size_ratio < threshold:
                continue
            score = len(left & right) / len(left | right)
            if score >= threshold:
                union(i, j)

    groups: Dict[int, List[str]] = {}
    for index, doc in enumerate(docs):
        groups.setdefault(find(index), []).append(str(doc["pmc_id"]))
    return sorted((sorted(g) for g in groups.values() if len(g) > 1), key=lambda g: g[0])


def create_balanced_kfold_split(
    docs: Sequence[Dict[str, Any]], *, k: int = 4, seed: int = 2026,
    linked_groups: Sequence[Sequence[str]] = (),
) -> Dict[str, Any]:
    """Balance a fixed document pool into k folds while keeping linked IDs together."""
    if k < 2 or len(docs) < k:
        raise ValueError("Need at least k documents and k >= 2")
    features = {str(d["pmc_id"]): document_features(d) for d in docs}
    parent = {pmc_id: pmc_id for pmc_id in features}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[b] = a

    for group in linked_groups:
        members = [str(x) for x in group if str(x) in features]
        for member in members[1:]:
            union(members[0], member)
    units: Dict[str, List[str]] = {}
    for pmc_id in features:
        units.setdefault(find(pmc_id), []).append(pmc_id)
    keys = list(next(iter(features.values())).keys())
    totals = {key: sum(item[key] for item in features.values()) for key in keys}
    scales = {key: max(totals[key] / k, 1) for key in keys}
    rng = random.Random(seed)
    tie = {root: rng.random() for root in units}

    def unit_features(root: str) -> Counter[str]:
        total: Counter[str] = Counter()
        for pmc_id in units[root]:
            total.update(features[pmc_id])
        return total

    sums = {root: unit_features(root) for root in units}
    ordered = sorted(units, key=lambda root: (sum(sums[root][key] / scales[key] for key in keys if key != "documents"), tie[root]), reverse=True)
    bins: List[List[str]] = [[] for _ in range(k)]
    bin_sums: List[Counter[str]] = [Counter() for _ in range(k)]
    target_size = len(docs) / k
    for root in ordered:
        def cost(index: int) -> Tuple[float, float, int]:
            balance = sum(((bin_sums[index][key] + sums[root][key]) / scales[key]) ** 2 for key in keys)
            size_penalty = ((len(bins[index]) + len(units[root])) / target_size) ** 4
            return balance + size_penalty, len(bins[index]), index
        selected = min(range(k), key=cost)
        bins[selected].extend(units[root]);bin_sums[selected].update(sums[root])
    pool = sorted(features)
    folds = [{"fold": index, "train": sorted(set(pool) - set(bucket)), "validation": sorted(bucket)} for index, bucket in enumerate(bins)]
    return {"seed": seed, "strategy": f"deterministic greedy balance over fixed pool into {k} folds", "development_pool": pool, "cv_folds": folds, "group_features": [dict(x) for x in bin_sums]}


def create_balanced_five_way_split(
    docs: Sequence[Dict[str, Any]], seed: int = 2026
) -> Dict[str, Any]:
    """Create one locked 20% split plus four folds over the remaining 80%."""
    if len(docs) % 5:
        raise ValueError("Document count must be divisible by five")
    capacity = len(docs) // 5
    features = {str(d["pmc_id"]): document_features(d) for d in docs}
    keys = list(next(iter(features.values())).keys())
    totals = {key: sum(item[key] for item in features.values()) for key in keys}
    scales = {key: max(totals[key] / 5, 1) for key in keys}
    rng = random.Random(seed)
    tie = {pmc_id: rng.random() for pmc_id in features}

    def rarity(pmc_id: str) -> Tuple[float, float]:
        item = features[pmc_id]
        return (sum(item[key] / scales[key] for key in keys if key != "documents"), tie[pmc_id])

    bins: List[List[str]] = [[] for _ in range(5)]
    bin_sums: List[Counter[str]] = [Counter() for _ in range(5)]
    for pmc_id in sorted(features, key=rarity, reverse=True):
        candidates = [i for i, bucket in enumerate(bins) if len(bucket) < capacity]

        def cost(index: int) -> Tuple[float, int, int]:
            score = sum(
                ((bin_sums[index][key] + features[pmc_id][key]) / scales[key]) ** 2
                for key in keys
            )
            return score, len(bins[index]), index

        selected = min(candidates, key=cost)
        bins[selected].append(pmc_id)
        bin_sums[selected].update(features[pmc_id])

    locked = sorted(bins[0])
    cv_validation = [sorted(bucket) for bucket in bins[1:]]
    development = sorted(x for bucket in bins[1:] for x in bucket)
    folds = [
        {
            "fold": index,
            "train": sorted(set(development) - set(validation)),
            "validation": validation,
        }
        for index, validation in enumerate(cv_validation)
    ]
    return {
        "seed": seed,
        "strategy": "deterministic greedy balance over five document groups",
        "locked_validation": locked,
        "development_pool": development,
        "cv_folds": folds,
        "group_features": [dict(bin_sums[i]) for i in range(5)],
    }


def write_e0_artifacts(data_dir: Path, output_dir: Path, seed: int = 2026) -> Dict[str, Path]:
    train_path = data_dir / "PatientPheX-train.jsonl"
    a_path = data_dir / "PatientPheX-A.jsonl"
    obo_path = data_dir / "hp.obo"
    docs = load_jsonl(str(train_path))
    ontology = HPOntology(str(obo_path))
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "files": {
            path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in [train_path, a_path, obo_path]
        },
        "hpo_data_version": next(
            (line.split(":", 1)[1].strip() for line in obo_path.read_text(encoding="utf-8").splitlines() if line.startswith("data-version:")),
            None,
        ),
    }
    audit = audit_dataset(docs, ontology)
    near_duplicates = find_near_duplicate_groups(docs)
    audit["near_duplicate_groups"] = near_duplicates
    split = create_balanced_five_way_split(docs, seed=seed)
    split["near_duplicate_groups"] = near_duplicates

    outputs = {
        "manifest": output_dir / "data_manifest.json",
        "audit": output_dir / "annotation_audit.json",
        "splits": output_dir / "splits.json",
    }
    for key, payload in [("manifest", manifest), ("audit", audit), ("splits", split)]:
        outputs[key].write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return outputs
