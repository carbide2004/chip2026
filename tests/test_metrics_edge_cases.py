"""Boundary tests derived from the published PatientPheX rules."""

import pytest

from src.metrics import evaluate_subtask1_mention, evaluate_subtask2


def _doc(entities=None, association=None):
    return {
        "pmc_id": "1",
        "entities": entities or [],
        "association": association or [],
    }


def _entity(identifier, offset=10, length=5, note=None):
    return {
        "identifier": identifier,
        "offset": offset,
        "length": length,
        "text": "x" * length,
        "note": note,
    }


def test_predicted_no_id_is_not_a_wildcard_for_valid_gold_id():
    gold = [_doc([_entity("HP:0000001")])]
    pred = [_doc([_entity("-1")])]

    result = evaluate_subtask1_mention(gold, pred)

    assert result["tp_men"] == 0
    assert result["fp_men"] == 1
    assert result["fn_men"] == 1


def test_gold_no_id_matches_on_exact_span_only():
    gold = [_doc([_entity("-1")])]
    pred = [_doc([_entity("HP:0000001")])]

    result = evaluate_subtask1_mention(gold, pred)

    assert result["tp_men"] == 1
    assert result["fp_men"] == 0
    assert result["fn_men"] == 0


def test_gold_no_id_can_match_only_one_component_of_composite_prediction():
    gold = [_doc([_entity("-1")])]
    pred = [_doc([_entity("HP:0000001;HP:0000002")])]

    result = evaluate_subtask1_mention(gold, pred)

    assert result["tp_men"] == 1
    assert result["fp_men"] == 1
    assert result["fn_men"] == 0


def test_composite_mentions_are_independent_evaluation_units():
    gold = [_doc([_entity("HP:0000001;HP:0000002")])]
    pred = [_doc([_entity("HP:0000001")])]

    result = evaluate_subtask1_mention(gold, pred)

    assert result["tp_men"] == 1
    assert result["fp_men"] == 0
    assert result["fn_men"] == 1


def test_empty_gold_and_prediction_patient_sets_score_one():
    gold = [_doc(association=[{"patient_id": "P1", "phenotype": []}])]
    pred = [_doc(association=[{"patient_id": "P1", "phenotype": []}])]

    result = evaluate_subtask2(gold, pred)

    assert result["f1_micro"] == 0.0
    assert result["f1_macro"] == pytest.approx(1.0)
