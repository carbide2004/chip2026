from src.submission import validate_submission


def _input_doc():
    return {
        "pmc_id": "1",
        "pmid": "2",
        "patient": [{"patient_id": "P1", "mention": []}],
        "full_text": [{"offset": 0, "text": "fever absent"}],
    }


def _prediction():
    return {
        "pmc_id": "1",
        "pmid": "2",
        "entities": [
            {
                "identifier": "HP:0001945",
                "type": "Phenotype",
                "offset": 0,
                "length": 5,
                "text": "fever",
                "note": None,
            }
        ],
        "association": [{"patient_id": "P1", "phenotype": ["HP:0001945"]}],
    }


def test_valid_submission_has_no_issues():
    assert validate_submission([_input_doc()], [_prediction()]) == []


def test_validator_detects_coordinate_and_patient_errors():
    prediction = _prediction()
    prediction["entities"][0]["offset"] = 1
    prediction["association"][0]["patient_id"] = "P2"

    codes = {issue.code for issue in validate_submission([_input_doc()], [prediction])}

    assert "entity_text_offset_mismatch" in codes
    assert "missing_patient" in codes
    assert "extra_patient" in codes
