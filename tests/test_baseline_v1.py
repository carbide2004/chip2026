from src.baseline_v1 import LexicalPhenotypeBaseline, associate_by_position
from src.data_loader import HPOntology


def _ontology(tmp_path):
    path = tmp_path / "hp.obo"
    path.write_text(
        """format-version: 1.2

[Term]
id: HP:0000118
name: Phenotypic abnormality

[Term]
id: HP:0000001
name: high fever
synonym: \"fever\" EXACT []
is_a: HP:0000118 ! Phenotypic abnormality
""",
        encoding="utf-8",
    )
    return HPOntology(str(path))


def test_lexical_baseline_preserves_offsets_and_detects_simple_negation(tmp_path):
    baseline = LexicalPhenotypeBaseline(_ontology(tmp_path))
    entities = baseline.extract_passage("Patient has no high fever.", 100, detect_negation=True)

    assert entities == [
        {
            "identifier": "HP:0000001",
            "type": "Phenotype",
            "offset": 115,
            "length": 10,
            "text": "high fever",
            "note": "NO",
        }
    ]


def test_association_baseline_does_not_special_case_single_patient():
    doc = {
        "patient": [{"patient_id": "P1", "mention": [{"offset": 10}]}],
        "full_text": [{"offset": 0, "text": "x" * 50}, {"offset": 1000, "text": "y" * 50}],
    }
    entities = [
        {"identifier": "HP:0000001", "offset": 1010, "note": None, "text": "x"}
    ]

    association = associate_by_position(doc, entities, method="window", window=100)

    assert association == [{"patient_id": "P1", "phenotype": []}]
