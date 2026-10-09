#!/usr/bin/env python3
"""Compare SapBERT dense retrieval and lexical+dense RRF on gold spans."""

from __future__ import annotations

from collections import defaultdict
import argparse
import json
from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch
from transformers import AutoModel, AutoTokenizer

from src.data_loader import HPOntology, load_jsonl
from src.hpo_retrieval import LexicalHPORetriever


def split_ids(raw):
    return [x.strip() for x in str(raw).split(";") if x.strip()]


def encode(texts, tokenizer, model, device, batch_size=256):
    output = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = tokenizer(
                texts[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=32,
                return_tensors="pt",
            ).to(device)
            hidden = model(**batch).last_hidden_state[:, 0]
            hidden = torch.nn.functional.normalize(hidden.float(), p=2, dim=1)
            output.append(hidden.cpu())
    return torch.cat(output)


def dense_ids(query_embeddings, phrase_embeddings, phrase_ids, top_phrases=200, top_ids=100, device="cuda"):
    phrase_gpu = phrase_embeddings.to(device)
    outputs = []
    for start in range(0, len(query_embeddings), 128):
        query = query_embeddings[start : start + 128].to(device)
        indices = (query @ phrase_gpu.T).topk(top_phrases, dim=1).indices.cpu().tolist()
        for row in indices:
            seen = set()
            ranked = []
            for phrase_index in row:
                for identifier in phrase_ids[phrase_index]:
                    if identifier not in seen:
                        seen.add(identifier)
                        ranked.append(identifier)
                        if len(ranked) == top_ids:
                            break
                if len(ranked) == top_ids:
                    break
            outputs.append(ranked)
    return outputs


def rrf(first, second, constant=60):
    scores = defaultdict(float)
    for ranking in [first, second]:
        for rank, identifier in enumerate(ranking, 1):
            scores[identifier] += 1 / (constant + rank)
    return [identifier for identifier, _ in sorted(scores.items(), key=lambda x: (-x[1], x[0]))]


def recall(records, rankings, method, k, selector=lambda _: True):
    selected = [r for r in records if selector(r)]
    hits = sum(r["identifier"] in rankings[(method, r["surface"])][:k] for r in selected)
    return {"count": len(selected), "hits": hits, "recall": hits / len(selected) if selected else None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data/raw/PatientPheX-V1-A")
    parser.add_argument("--e0-dir", type=Path, default=REPO_ROOT / "data/processed/e0")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "data/processed/e2")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    docs = load_jsonl(str(args.data_dir / "PatientPheX-train.jsonl"))
    by_id = {str(d["pmc_id"]): d for d in docs}
    split = json.loads((args.e0_dir / "splits.json").read_text())
    development_ids = set(split["development_pool"])
    ontology = HPOntology(str(args.data_dir / "hp.obo"))
    lexical = LexicalHPORetriever(ontology)

    records, excluded = [], defaultdict(int)
    for doc in docs:
        if str(doc["pmc_id"]) not in development_ids:
            continue
        for entity in doc.get("entities", []):
            if str(entity.get("note") or "").upper() == "NO":
                continue
            surface = lexical.normalize(entity.get("text", ""))
            for raw_id in split_ids(entity.get("identifier")):
                if raw_id == "-1":
                    excluded["no_id"] += 1
                    continue
                identifier = ontology.resolve_id(raw_id)
                if identifier is None:
                    excluded["unresolvable"] += 1
                elif not ontology.terms[identifier].is_sub_abnormality:
                    excluded["outside_branch"] += 1
                else:
                    records.append(
                        {
                            "pmc_id": str(doc["pmc_id"]),
                            "surface": surface,
                            "identifier": identifier,
                            "exact_for_gold": identifier in lexical.phrase_to_ids.get(surface, set()),
                        }
                    )
    surfaces = sorted({r["surface"] for r in records})
    phrases = lexical.phrases
    phrase_ids = [sorted(lexical.phrase_to_ids[p]) for p in phrases]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device)
    revision = getattr(model.config, "_commit_hash", None)
    parameter_count = sum(p.numel() for p in model.parameters())
    phrase_embeddings = encode(phrases, tokenizer, model, device)
    query_embeddings = encode(surfaces, tokenizer, model, device)
    dense = dense_ids(query_embeddings, phrase_embeddings, phrase_ids, device=device)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    rankings = {}
    for surface, dense_ranking in zip(surfaces, dense):
        lexical_ranking = [identifier for identifier, _ in lexical.retrieve(surface, scorer="token_set", top_k=100)]
        rankings[("dense", surface)] = dense_ranking
        rankings[("lexical_token_set", surface)] = lexical_ranking
        rankings[("rrf", surface)] = rrf(dense_ranking, lexical_ranking)

    results = []
    for method in ["dense", "lexical_token_set", "rrf"]:
        for k in [1, 5, 10, 20]:
            row = {
                "method": method,
                "k": k,
                "overall": recall(records, rankings, method, k),
                "exact_surface": recall(records, rankings, method, k, lambda r: r["exact_for_gold"]),
                "non_exact_surface": recall(records, rankings, method, k, lambda r: not r["exact_for_gold"]),
                "folds": [],
            }
            for fold in split["cv_folds"]:
                train_ids, validation_ids = set(fold["train"]), set(fold["validation"])
                seen = set()
                for pmc_id in train_ids:
                    for entity in by_id[pmc_id].get("entities", []):
                        if str(entity.get("note") or "").upper() == "NO":
                            continue
                        for raw_id in split_ids(entity.get("identifier")):
                            resolved = ontology.resolve_id(raw_id)
                            if resolved:
                                seen.add(resolved)
                fold_records = [r for r in records if r["pmc_id"] in validation_ids]
                row["folds"].append(
                    {
                        "fold": fold["fold"],
                        "overall": recall(fold_records, rankings, method, k),
                        "seen_id": recall(fold_records, rankings, method, k, lambda r, s=seen: r["identifier"] in s),
                        "unseen_id": recall(fold_records, rankings, method, k, lambda r, s=seen: r["identifier"] not in s),
                    }
                )
            results.append(row)

    report = {
        "scope": "gold positive mention spans in 64-document development pool; locked validation untouched",
        "model": {
            "name": args.model,
            "revision": revision,
            "parameter_count": parameter_count,
            "pooling": "CLS with L2 normalization",
            "max_length": 32,
        },
        "records": len(records),
        "excluded": dict(excluded),
        "elapsed_seconds": time.time() - started,
        "results": results,
    }
    path = args.output_dir / "dense_retrieval_report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"model": report["model"], "elapsed_seconds": report["elapsed_seconds"], "results": results}, ensure_ascii=False, indent=2))
    print(f"report: {path}")


if __name__ == "__main__":
    main()
