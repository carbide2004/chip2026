"""Full dictionary-based entity extraction using AC Automaton / multi-term matching.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple
import re

from src.data_loader import HPOntology


@dataclass
class ExtractedEntity:
    identifier: str
    type: str  # "Phenotype"
    offset: int
    length: int
    text: str
    note: Optional[str] = None
    context_sentence: str = ""


class DictionaryExtractor:
    """Multi-term maximal phrase matching across full HPO dictionary."""

    def __init__(self, ontology: HPOntology):
        self.ontology = ontology
        # Phrase -> HPO IDs mapping
        self.phrase_to_ids: Dict[str, Set[str]] = {}
        for phrase, ids in self.ontology.text_to_ids.items():
            clean = phrase.strip().lower()
            if len(clean) >= 3 and not clean.isdigit():
                self.phrase_to_ids[clean] = ids

        # Token count buckets to speed up search without giant regex
        # Build token-level inverted index
        self.first_word_index: Dict[str, List[str]] = {}
        for p in self.phrase_to_ids.keys():
            first_w = p.split()[0]
            self.first_word_index.setdefault(first_w, []).append(p)

        # Sort each bucket by length descending
        for w in self.first_word_index:
            self.first_word_index[w].sort(key=len, reverse=True)

    def extract_from_text(
        self, text: str, global_offset: int = 0
    ) -> List[ExtractedEntity]:
        """Token-guided maximal matching against all HPO phrases."""
        entities: List[ExtractedEntity] = []
        lower_text = text.lower()
        n = len(text)

        # Tokenize by word boundaries: find start positions of words
        word_iter = re.finditer(r"\b[a-z0-9\-]+\b", lower_text)
        occupied_spans: List[Tuple[int, int]] = []

        for m in word_iter:
            start = m.start()
            word = m.group(0)

            # If this position is already inside an accepted maximal entity span, skip
            if any(s <= start < e for s, e in occupied_spans):
                continue

            candidates = self.first_word_index.get(word)
            if not candidates:
                continue

            # Check candidate phrases starting with this word
            for phrase in candidates:
                plen = len(phrase)
                if start + plen <= n:
                    if lower_text[start : start + plen] == phrase:
                        # Ensure right word boundary
                        if (
                            start + plen == n
                            or not lower_text[start + plen].isalnum()
                        ):
                            # Matched maximal phrase!
                            end = start + plen
                            matched_text = text[start:end]
                            ids = sorted(list(self.phrase_to_ids[phrase]))
                            identifier = ";".join(ids[:1])

                            ctx_start = max(0, start - 100)
                            ctx_end = min(n, end + 100)
                            context = (
                                text[ctx_start:ctx_end]
                                .replace("\n", " ")
                                .strip()
                            )

                            occupied_spans.append((start, end))
                            entities.append(
                                ExtractedEntity(
                                    identifier=identifier,
                                    type="Phenotype",
                                    offset=global_offset + start,
                                    length=plen,
                                    text=matched_text,
                                    note=None,
                                    context_sentence=context,
                                )
                            )
                            break

        entities.sort(key=lambda x: x.offset)
        return entities

    def extract_from_doc(self, doc: Dict[str, Any]) -> List[ExtractedEntity]:
        all_entities: List[ExtractedEntity] = []
        for sec in doc.get("full_text", []):
            sec_text = sec.get("text", "")
            sec_offset = sec.get("offset", 0)
            sec_type = (sec.get("section_type") or "").upper()

            if any(
                skip in sec_type
                for skip in ["REF", "ACK", "FOOTNOTE", "SUPPL", "AUTHOR"]
            ):
                continue

            sec_ents = self.extract_from_text(
                sec_text, global_offset=sec_offset
            )
            all_entities.extend(sec_ents)

        return all_entities
