"""Unit tests for PatientPheX metrics and data loaders.
"""

from pathlib import Path
import sys
import pytest

# Ensure src is importable
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from src.data_loader import HPOntology, load_jsonl, save_jsonl
from src.metrics import compute_all_metrics, evaluate_subtask1_mention, evaluate_subtask1_doc, evaluate_subtask2


def test_metrics_perfect_match():
    gold = [
        {
            "pmc_id": "PMC001",
            "entities": [
                {"identifier": "HP:0001234", "offset": 10, "length": 5, "text": "fever", "note": None},
                {"identifier": "HP:0005678", "offset": 30, "length": 4, "text": "pain", "note": "NO"},  # Negation
            ],
            "association": [
                {"patient_id": "P1", "phenotype": ["HP:0001234"]}
            ]
        }
    ]

    # Exactly matching prediction
    pred = [
        {
            "pmc_id": "PMC001",
            "entities": [
                {"identifier": "HP:0001234", "offset": 10, "length": 5, "text": "fever", "note": None},
                {"identifier": "HP:0005678", "offset": 30, "length": 4, "text": "pain", "note": "NO"},
            ],
            "association": [
                {"patient_id": "P1", "phenotype": ["HP:0001234"]}
            ]
        }
    ]

    res = compute_all_metrics(gold, pred)
    assert res["overall_score"] == pytest.approx(1.0, 1e-5)
    assert res["subtask1"]["f1_men"] == pytest.approx(1.0, 1e-5)
    assert res["subtask1"]["f1_doc"] == pytest.approx(1.0, 1e-5)
    assert res["subtask2"]["f1_micro"] == pytest.approx(1.0, 1e-5)
    assert res["subtask2"]["f1_macro"] == pytest.approx(1.0, 1e-5)


def test_metrics_negation_handling():
    gold = [
        {
            "pmc_id": "PMC001",
            "entities": [
                {"identifier": "HP:0005678", "offset": 30, "length": 4, "text": "pain", "note": "NO"},
            ],
            "association": []
        }
    ]

    # Predict negation as regular phenotype (must count as FP)
    pred_bad = [
        {
            "pmc_id": "PMC001",
            "entities": [
                {"identifier": "HP:0005678", "offset": 30, "length": 4, "text": "pain", "note": None},
            ],
            "association": []
        }
    ]

    res = compute_all_metrics(gold, pred_bad)
    # Predicted 1 false positive, gold has 0 positives
    assert res["subtask1"]["fp_men"] == 1
    assert res["subtask1"]["f1_men"] == 0.0


def test_data_loader_and_hpo():
    obo_path = repo_root / "data" / "raw" / "PatientPheX-V1-A" / "hp.obo"
    if not obo_path.exists():
        pytest.skip("hp.obo not found")

    hpo = HPOntology(str(obo_path))
    assert len(hpo.terms) > 10000

    # HP:0000118 is the root of phenotypic abnormality
    assert "HP:0000118" in hpo.terms
    assert hpo.terms["HP:0000118"].is_sub_abnormality

    # Look up a common term
    res = hpo.lookup_exact("fever")
    assert len(res) > 0
    assert any(hpo.terms[tid].is_sub_abnormality for tid in res)
