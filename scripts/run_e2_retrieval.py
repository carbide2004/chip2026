#!/usr/bin/env python3
"""Evaluate HPO retrieval with gold mention spans on CV folds only."""

from __future__ import annotations

from collections import defaultdict
import argparse
import json
from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data_loader import HPOntology, load_jsonl
from src.hpo_retrieval import LexicalHPORetriever


def identifiers(raw):
    return [x.strip() for x in str(raw).split(";") if x.strip()]


def positive_ids(doc, ontology):
    result = set()
    for entity in doc.get("entities", []):
        if str(entity.get("note") or "").upper() == "NO":
            continue
        for identifier in identifiers(entity.get("identifier")):
            resolved = ontology.resolve_id(identifier)
            if resolved and ontology.terms[resolved].is_sub_abnormality:
                result.add(resolved)
    return result


def summarize(records, scorer, k, retrievals, selector=lambda _: True):
    chosen = [record for record in records if selector(record)]
    hits = sum(record["identifier"] in retrievals[(scorer, record["surface"])][:k] for record in chosen)
    return {"count": len(chosen), "hits": hits, "recall": hits / len(chosen) if chosen else None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data/raw/PatientPheX-V1-A")
    parser.add_argument("--e0-dir", type=Path, default=REPO_ROOT / "data/processed/e0")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "data/processed/e2")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    docs = load_jsonl(str(args.data_dir / "PatientPheX-train.jsonl"))
    by_id = {str(doc["pmc_id"]): doc for doc in docs}
    split = json.loads((args.e0_dir / "splits.json").read_text(encoding="utf-8"))
    development_ids = set(split["development_pool"])
    ontology = HPOntology(str(args.data_dir / "hp.obo"))
    retriever = LexicalHPORetriever(ontology)

    records = []
    excluded = defaultdict(int)
    for doc in docs:
        if str(doc["pmc_id"]) not in development_ids:
            continue
        for entity in doc.get("entities", []):
            if str(entity.get("note") or "").upper() == "NO":
                continue
            surface = retriever.normalize(entity.get("text", ""))
            for original_id in identifiers(entity.get("identifier")):
                if original_id == "-1":
                    excluded["no_id"] += 1
                    continue
                resolved = ontology.resolve_id(original_id)
                if resolved is None:
                    excluded["unresolvable"] += 1
                    continue
                if not ontology.terms[resolved].is_sub_abnormality:
                    excluded["outside_branch"] += 1
                    continue
                records.append(
                    {
                        "pmc_id": str(doc["pmc_id"]),
                        "surface": surface,
                        "identifier": resolved,
                        "original_identifier": original_id,
                        "exact_for_gold": resolved in retriever.phrase_to_ids.get(surface, set()),
                    }
                )

    scorers = ["ratio", "wratio", "token_set"]
    surfaces = sorted({record["surface"] for record in records})
    retrievals = {}
    started = time.time()
    for scorer in scorers:
        for index, surface in enumerate(surfaces, 1):
            retrievals[(scorer, surface)] = [
                identifier for identifier, _ in retriever.retrieve(surface, scorer=scorer, top_k=20)
            ]
            if index % 500 == 0:
                print(f"{scorer}: {index}/{len(surfaces)}", flush=True)

    results = []
    for scorer in scorers:
        for k in [1, 5, 10, 20]:
            row = {
                "scorer": scorer,
                "k": k,
                "overall": summarize(records, scorer, k, retrievals),
                "exact_surface": summarize(records, scorer, k, retrievals, lambda r: r["exact_for_gold"]),
                "non_exact_surface": summarize(records, scorer, k, retrievals, lambda r: not r["exact_for_gold"]),
                "folds": [],
            }
            for fold in split["cv_folds"]:
                train_ids = set(fold["train"])
                validation_ids = set(fold["validation"])
                seen = set().union(*(positive_ids(by_id[x], ontology) for x in train_ids))
                fold_records = [record for record in records if record["pmc_id"] in validation_ids]
                row["folds"].append(
                    {
                        "fold": fold["fold"],
                        "overall": summarize(fold_records, scorer, k, retrievals),
                        "seen_id": summarize(fold_records, scorer, k, retrievals, lambda r, s=seen: r["identifier"] in s),
                        "unseen_id": summarize(fold_records, scorer, k, retrievals, lambda r, s=seen: r["identifier"] not in s),
                    }
                )
            results.append(row)

    results.sort(key=lambda row: (row["overall"]["recall"], -row["k"]), reverse=True)
    report = {
        "scope": "gold positive mention spans in 64-document development pool; locked validation untouched",
        "unit": "one gold HPO identifier after resolvable alt/obsolete normalization",
        "excluded": dict(excluded),
        "record_count": len(records),
        "unique_surface_count": len(surfaces),
        "elapsed_seconds": time.time() - started,
        "results": results,
    }
    path = args.output_dir / "retrieval_report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"records": len(records), "excluded": dict(excluded), "top": results[:8]}, ensure_ascii=False, indent=2))
    print(f"report: {path}")


if __name__ == "__main__":
    main()
