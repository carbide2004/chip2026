#!/usr/bin/env python3
"""Validate a PatientPheX prediction JSONL before submission."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data_loader import load_jsonl
from src.submission import validate_submission


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="Official A/B input JSONL")
    parser.add_argument("prediction", type=Path, help="Prediction JSONL")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    issues = validate_submission(load_jsonl(str(args.input)), load_jsonl(str(args.prediction)))
    counts = Counter((item.severity, item.code) for item in issues)
    summary = {
        "errors": sum(item.severity == "error" for item in issues),
        "warnings": sum(item.severity == "warning" for item in issues),
        "counts": {f"{severity}:{code}": count for (severity, code), count in sorted(counts.items())},
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps([item.to_dict() for item in issues], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    raise SystemExit(1 if summary["errors"] else 0)


if __name__ == "__main__":
    main()
