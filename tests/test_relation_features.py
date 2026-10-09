from src.relation_features import build_examples, feature_names


def test_relation_examples_are_patient_concept_bags_not_mentions():
    doc = {
        "pmc_id": "1",
        "patient": [
            {"patient_id": "P1", "mention": [{"offset": 5}]},
            {"patient_id": "P2", "mention": [{"offset": 105}]},
        ],
        "full_text": [{"offset": 0, "text": "x" * 200, "section_type": "CASE"}],
        "entities": [
            {"identifier": "HP:0000001", "offset": 10, "text": "a", "note": None},
            {"identifier": "HP:0000001", "offset": 110, "text": "a", "note": None},
        ],
        "association": [
            {"patient_id": "P1", "phenotype": ["HP:0000001"]},
            {"patient_id": "P2", "phenotype": []},
        ],
    }

    features, labels, keys = build_examples([doc])

    assert len(features) == 2
    assert labels == [1, 0]
    assert keys == [("1", "P1", "HP:0000001"), ("1", "P2", "HP:0000001")]
    assert all(len(row) == len(feature_names()) for row in features)
