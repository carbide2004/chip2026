#!/usr/bin/env python3
"""Run reproducible E1 lexical and positional baselines on CV folds only."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import statistics
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.baseline_v1 import LexicalPhenotypeBaseline, build_prediction
from src.data_loader import HPOntology, load_jsonl, save_jsonl
from src.metrics import compute_all_metrics


def subset(docs, ids):
    wanted = set(ids)
    return [doc for doc in docs if str(doc["pmc_id"]) in wanted]


def mean_components(folds):
    paths = {
        "overall_score": lambda x: x["overall_score"],
        "f1_men": lambda x: x["subtask1"]["f1_men"],
        "f1_doc": lambda x: x["subtask1"]["f1_doc"],
        "f1_micro": lambda x: x["subtask2"]["f1_micro"],
        "f1_macro": lambda x: x["subtask2"]["f1_macro"],
    }
    return {key: statistics.mean(fn(value) for value in folds) for key, fn in paths.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data/raw/PatientPheX-V1-A")
    parser.add_argument("--e0-dir", type=Path, default=REPO_ROOT / "data/processed/e0")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "data/processed/e1")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    docs = load_jsonl(str(args.data_dir / "PatientPheX-train.jsonl"))
    by_id = {str(doc["pmc_id"]): doc for doc in docs}
    splits = json.loads((args.e0_dir / "splits.json").read_text(encoding="utf-8"))
    development = subset(docs, splits["development_pool"])
    ontology = HPOntology(str(args.data_dir / "hp.obo"))
    extractor = LexicalPhenotypeBaseline(ontology)

    entity_configs = [
        {"overlap_policy": overlap, "detect_negation": negation}
        for overlap in ["longest", "all"]
        for negation in [False, True]
    ]
    association_configs = [
        {"method": "all", "window": 0},
        {"method": "same_passage", "window": 0},
        *[{"method": "window", "window": value} for value in [250, 500, 1000, 1500]],
        *[{"method": "nearest", "window": value} for value in [500, 1500, 10**9]],
    ]

    started = time.time()
    entity_cache = {}
    for config in entity_configs:
        key = f"{config['overlap_policy']}-neg{int(config['detect_negation'])}"
        entity_cache[key] = {
            str(doc["pmc_id"]): extractor.extract_document(doc, **config) for doc in development
        }

    runs = []
    for entity_config in entity_configs:
        entity_key = f"{entity_config['overlap_policy']}-neg{int(entity_config['detect_negation'])}"
        for association_config in association_configs:
            name = f"lex-{entity_key}__{association_config['method']}-{association_config['window']}"
            fold_metrics = []
            for fold in splits["cv_folds"]:
                gold = subset(docs, fold["validation"])
                predictions = [
                    build_prediction(
                        doc,
                        entity_cache[entity_key][str(doc["pmc_id"])],
                        association_method=association_config["method"],
                        association_window=association_config["window"],
                    )
                    for doc in gold
                ]
                fold_metrics.append(compute_all_metrics(gold, predictions))
            runs.append(
                {
                    "name": name,
                    "entity_config": entity_config,
                    "association_config": association_config,
                    "mean": mean_components(fold_metrics),
                    "folds": fold_metrics,
                }
            )

    oracle_runs = []
    for association_config in association_configs:
        fold_metrics = []
        for fold in splits["cv_folds"]:
            gold = subset(docs, fold["validation"])
            predictions = [
                build_prediction(
                    doc,
                    doc.get("entities", []),
                    association_method=association_config["method"],
                    association_window=association_config["window"],
                )
                for doc in gold
            ]
            fold_metrics.append(compute_all_metrics(gold, predictions))
        oracle_runs.append(
            {
                "association_config": association_config,
                "mean": mean_components(fold_metrics),
                "folds": fold_metrics,
            }
        )

    runs.sort(key=lambda item: item["mean"]["overall_score"], reverse=True)
    oracle_runs.sort(
        key=lambda item: item["mean"]["f1_micro"] + item["mean"]["f1_macro"], reverse=True
    )
    report = {
        "scope": "64-document development pool; locked validation untouched",
        "association_composite_semantics": "provisional: src.metrics splits semicolon strings",
        "elapsed_seconds": time.time() - started,
        "runs": runs,
        "gold_entity_relation_oracle_runs": oracle_runs,
    }
    report_path = args.output_dir / "baseline_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    best = runs[0]
    entity_key = f"{best['entity_config']['overlap_policy']}-neg{int(best['entity_config']['detect_negation'])}"
    best_predictions = [
        build_prediction(
            doc,
            entity_cache[entity_key][str(doc["pmc_id"])],
            association_method=best["association_config"]["method"],
            association_window=best["association_config"]["window"],
        )
        for doc in development
    ]
    save_jsonl(best_predictions, str(args.output_dir / "best_development_predictions.jsonl"))
    print(json.dumps({"best": best["name"], "mean": best["mean"], "gold_entity_relation_best": oracle_runs[0]}, ensure_ascii=False, indent=2))
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
