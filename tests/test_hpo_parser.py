from src.data_loader import HPOntology


def test_hpo_parser_preserves_alias_and_obsolescence_metadata(tmp_path):
    obo = tmp_path / "test.obo"
    obo.write_text(
        """format-version: 1.2

[Term]
id: HP:0000118
name: Phenotypic abnormality

[Term]
id: HP:0000001
name: Active term
alt_id: HP:0099999
synonym: \"Exact alias\" EXACT []
synonym: \"Broad alias\" BROAD []
is_a: HP:0000118 ! Phenotypic abnormality

[Term]
id: HP:0000002
name: Old term
is_obsolete: true
replaced_by: HP:0000001
consider: HP:0000003
""",
        encoding="utf-8",
    )

    ontology = HPOntology(str(obo))

    term = ontology.terms["HP:0000001"]
    assert term.alt_ids == ["HP:0099999"]
    assert term.synonyms[0] == "Exact alias"
    assert term.synonym_scopes["Exact alias"] == "EXACT"
    assert ontology.resolve_id("HP:0099999") == "HP:0000001"
    assert ontology.lookup_exact("Exact alias") == {"HP:0000001"}

    obsolete = ontology.terms["HP:0000002"]
    assert obsolete.replaced_by == ["HP:0000001"]
    assert obsolete.consider == ["HP:0000003"]
    assert ontology.resolve_id("HP:0000002") == "HP:0000001"
