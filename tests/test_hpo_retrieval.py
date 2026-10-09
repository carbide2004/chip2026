from src.data_loader import HPOntology
from src.hpo_retrieval import LexicalHPORetriever


def test_retrieval_returns_exact_and_fuzzy_candidates(tmp_path):
    path = tmp_path / "hp.obo"
    path.write_text(
        """format-version: 1.2

[Term]
id: HP:0000118
name: Phenotypic abnormality

[Term]
id: HP:0000001
name: high fever
synonym: \"pyrexia\" EXACT []
is_a: HP:0000118 ! Phenotypic abnormality

[Term]
id: HP:0000002
name: severe pain
is_a: HP:0000118 ! Phenotypic abnormality
""",
        encoding="utf-8",
    )
    retriever = LexicalHPORetriever(HPOntology(str(path)))

    assert retriever.retrieve("pyrexia", top_k=1)[0][0] == "HP:0000001"
    assert retriever.retrieve("high fevr", scorer="ratio", top_k=1)[0][0] == "HP:0000001"
