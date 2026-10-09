"""HPO candidate retrieval with transparent lexical baselines."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Set, Tuple

from rapidfuzz import fuzz, process

from .data_loader import HPOntology


class LexicalHPORetriever:
    def __init__(self, ontology: HPOntology):
        self.ontology = ontology
        phrase_to_ids: Dict[str, Set[str]] = defaultdict(set)
        self.canonical_names: Dict[str, str] = {}
        for identifier, term in ontology.terms.items():
            if term.is_obsolete or not term.is_sub_abnormality:
                continue
            self.canonical_names[identifier] = self.normalize(term.name)
            for phrase in [term.name, *term.synonyms]:
                normalized = self.normalize(phrase)
                if normalized:
                    phrase_to_ids[normalized].add(identifier)
        self.phrase_to_ids = dict(phrase_to_ids)
        self.phrases = list(self.phrase_to_ids)

    @staticmethod
    def normalize(text: str) -> str:
        return " ".join(text.lower().split())

    def retrieve(self, text: str, *, scorer: str = "wratio", top_k: int = 20) -> List[Tuple[str, float]]:
        scorers = {
            "ratio": fuzz.ratio,
            "wratio": fuzz.WRatio,
            "token_set": fuzz.token_set_ratio,
        }
        if scorer not in scorers:
            raise ValueError(f"Unknown scorer: {scorer}")
        query = self.normalize(text)
        phrase_limit = min(max(top_k * 10, 100), len(self.phrases))
        matches = process.extract(query, self.phrases, scorer=scorers[scorer], limit=phrase_limit)
        id_scores: Dict[str, float] = {}
        for phrase, score, _ in matches:
            for identifier in self.phrase_to_ids[phrase]:
                adjusted = float(score)
                if phrase == query and self.canonical_names.get(identifier) == query:
                    adjusted += 0.01
                id_scores[identifier] = max(id_scores.get(identifier, float("-inf")), adjusted)
        return sorted(id_scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]
