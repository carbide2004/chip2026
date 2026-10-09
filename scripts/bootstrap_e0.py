#!/usr/bin/env python3
"""Build reproducible E0 audit, manifest, and document splits."""

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.e0 import write_e0_artifacts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT / "data" / "raw" / "PatientPheX-V1-A",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "e0",
    )
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    outputs = write_e0_artifacts(args.data_dir, args.output_dir, seed=args.seed)
    audit = json.loads(outputs["audit"].read_text(encoding="utf-8"))
    splits = json.loads(outputs["splits"].read_text(encoding="utf-8"))
    print(f"Wrote E0 artifacts to {args.output_dir}")
    print(json.dumps(audit["summary"], ensure_ascii=False, indent=2))
    print(
        "split sizes:",
        len(splits["development_pool"]),
        "development /",
        len(splits["locked_validation"]),
        "locked /",
        [len(f["validation"]) for f in splits["cv_folds"]],
        "cv validation",
    )


if __name__ == "__main__":
    main()
