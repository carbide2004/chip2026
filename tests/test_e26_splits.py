from src.e0 import create_balanced_kfold_split


def _doc(index):
    return {
        "pmc_id": str(index),
        "full_text": [{"text": "case text " * (index + 1), "offset": 0}],
        "patient": [{"patient_id": "P1", "mention": []}],
        "entities": [],
        "association": [{"patient_id": "P1", "phenotype": []}],
    }


def test_repeated_split_is_complete_disjoint_and_deterministic():
    docs = [_doc(i) for i in range(12)]
    first = create_balanced_kfold_split(docs, k=4, seed=7, linked_groups=[["0", "1"]])
    second = create_balanced_kfold_split(docs, k=4, seed=7, linked_groups=[["0", "1"]])
    assert first == second
    validation = [set(fold["validation"]) for fold in first["cv_folds"]]
    assert set.union(*validation) == {str(i) for i in range(12)}
    assert sum(map(len, validation)) == 12
    assert all(validation[i].isdisjoint(validation[j]) for i in range(4) for j in range(i))
    assert any({"0", "1"}.issubset(fold) for fold in validation)
    for fold in first["cv_folds"]:
        assert set(fold["train"]).isdisjoint(fold["validation"])
