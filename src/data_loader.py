"""Data parsing and loading utilities for PatientPheX dataset and HPO ontology.
"""

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple


@dataclass
class HPOTerm:
    id: str
    name: str
    synonyms: List[str] = field(default_factory=list)
    synonym_scopes: Dict[str, str] = field(default_factory=dict)
    alt_ids: List[str] = field(default_factory=list)
    is_a: List[str] = field(default_factory=list)
    definition: str = ""
    is_obsolete: bool = False
    replaced_by: List[str] = field(default_factory=list)
    consider: List[str] = field(default_factory=list)
    is_sub_abnormality: bool = False


class HPOntology:
    """Parser and query index for hp.obo ontology."""

    def __init__(self, obo_path: str):
        self.obo_path = Path(obo_path)
        self.terms: Dict[str, HPOTerm] = {}
        self.alt_id_to_id: Dict[str, str] = {}
        self.text_to_ids: Dict[str, Set[str]] = {}
        self._load_obo()
        self._filter_phenotypic_abnormality()
        self._build_index()

    def _load_obo(self) -> None:
        """Parse OBO format into HPOTerm objects."""
        current_term: Optional[Dict[str, Any]] = None

        with open(self.obo_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line == "[Term]":
                    if current_term and "id" in current_term:
                        self._save_term(current_term)
                    current_term = {
                        "synonyms": [],
                        "synonym_scopes": {},
                        "alt_ids": [],
                        "is_a": [],
                        "is_obsolete": False,
                        "replaced_by": [],
                        "consider": [],
                    }
                elif line.startswith("[") and line.endswith("]"):
                    # Another stanza type like [Typedef]
                    if current_term and "id" in current_term:
                        self._save_term(current_term)
                    current_term = None
                elif current_term is not None:
                    if line.startswith("id:"):
                        current_term["id"] = line[3:].strip()
                    elif line.startswith("name:"):
                        current_term["name"] = line[5:].strip()
                    elif line.startswith("def:"):
                        match = re.search(r'"([^"]*)"', line[4:])
                        if match:
                            current_term["definition"] = match.group(1)
                    elif line.startswith("synonym:"):
                        match = re.search(r'^synonym:\s+"([^"]*)"\s+(EXACT|BROAD|NARROW|RELATED)', line)
                        if match:
                            text, scope = match.groups()
                            current_term["synonyms"].append(text)
                            current_term["synonym_scopes"][text] = scope
                    elif line.startswith("alt_id:"):
                        current_term["alt_ids"].append(line[7:].strip())
                    elif line.startswith("is_a:"):
                        parent_id = line[5:].split("!")[0].strip()
                        current_term["is_a"].append(parent_id)
                    elif line.startswith("is_obsolete:"):
                        if line[12:].strip().lower() == "true":
                            current_term["is_obsolete"] = True
                    elif line.startswith("replaced_by:"):
                        current_term["replaced_by"].append(line[12:].strip())
                    elif line.startswith("consider:"):
                        current_term["consider"].append(line[9:].strip())

        if current_term and "id" in current_term:
            self._save_term(current_term)

    def _save_term(self, d: Dict[str, Any]) -> None:
        term = HPOTerm(
            id=d["id"],
            name=d.get("name", ""),
            synonyms=d.get("synonyms", []),
            synonym_scopes=d.get("synonym_scopes", {}),
            alt_ids=d.get("alt_ids", []),
            is_a=d.get("is_a", []),
            definition=d.get("definition", ""),
            is_obsolete=d.get("is_obsolete", False),
            replaced_by=d.get("replaced_by", []),
            consider=d.get("consider", []),
        )
        self.terms[term.id] = term
        for alt_id in term.alt_ids:
            self.alt_id_to_id[alt_id] = term.id

    def _filter_phenotypic_abnormality(self) -> None:
        """Mark terms belonging to Phenotypic abnormality (HP:0000118) branch."""
        root_id = "HP:0000118"

        # Breadth-first / DFS search for descendants
        # Build children graph
        children_map: Dict[str, List[str]] = {}
        for term_id, term in self.terms.items():
            for p in term.is_a:
                children_map.setdefault(p, []).append(term_id)

        queue = [root_id]
        visited: Set[str] = set()

        while queue:
            curr = queue.pop(0)
            if curr in visited:
                continue
            visited.add(curr)
            if curr in self.terms:
                self.terms[curr].is_sub_abnormality = True
            for child in children_map.get(curr, []):
                if child not in visited:
                    queue.append(child)

    def _build_index(self) -> None:
        """Build exact lowercase text match index for terms under HP:0000118."""
        for term_id, term in self.terms.items():
            if term.is_obsolete or not term.is_sub_abnormality:
                continue
            # index main name
            if term.name:
                key = term.name.strip().lower()
                self.text_to_ids.setdefault(key, set()).add(term_id)
            # index synonyms
            for syn in term.synonyms:
                key = syn.strip().lower()
                self.text_to_ids.setdefault(key, set()).add(term_id)

    def lookup_exact(self, text: str) -> Set[str]:
        """Lookup active in-branch HPO IDs matching lowercase text exactly."""
        return self.text_to_ids.get(text.strip().lower(), set())

    def resolve_id(self, identifier: str) -> Optional[str]:
        """Resolve an alternative or obsolete HPO ID when OBO gives one answer.

        Ambiguous ``consider`` suggestions are intentionally not auto-resolved.
        """
        identifier = self.alt_id_to_id.get(identifier, identifier)
        term = self.terms.get(identifier)
        if term is None:
            return None
        if term.is_obsolete:
            return term.replaced_by[0] if len(term.replaced_by) == 1 else None
        return identifier


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load JSONL dataset file into a list of dictionaries."""
    docs = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                docs.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse line {line_idx} in {file_path}: {e}")
    return docs


def save_jsonl(docs: List[Dict[str, Any]], file_path: str) -> None:
    """Save document predictions to a UTF-8 JSONL file adhering to submission format."""
    out_path = Path(file_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for doc in docs:
            # Ensure proper key structure
            cleaned_doc = {
                "pmc_id": doc.get("pmc_id", ""),
                "pmid": doc.get("pmid"),
                "entities": doc.get("entities", []),
                "association": doc.get("association", []),
            }
            f.write(json.dumps(cleaned_doc, ensure_ascii=False) + "\n")
