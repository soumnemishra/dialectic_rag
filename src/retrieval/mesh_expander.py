import logging
import re
from typing import List
from pathlib import Path

from src.utils.debug_utils import get_debug_manager

logger = logging.getLogger(__name__)


class MeshExpander:
    """MeSH/UMLS expansion helper.

    Tries to use Biopython Entrez when available to look up MeSH headings.
    Falls back to lightweight heuristics (abbreviations, simple synonyms).
    """

    def __init__(self, email: str = None, api_key: str = None):
        self.email = email
        self.api_key = api_key
        try:
            from Bio import Entrez
            Entrez.email = email or Entrez.email
            if api_key:
                Entrez.api_key = api_key
            self.Entrez = Entrez
            self.has_entrez = True
        except Exception:
            self.Entrez = None
            self.has_entrez = False

    def _simple_synonyms(self, term: str) -> List[str]:
        # Very small heuristic mapping; users can extend
        mapping = {
            "myocardial infarction": ["heart attack"],
            "hypertension": ["high blood pressure"],
            "non-small cell lung cancer": ["NSCLC"],
        }
        res = [term]
        term_l = term.lower()
        if term_l in mapping:
            res.extend(mapping[term_l])

        # extract abbreviation in parentheses: "acute coronary syndrome (ACS)"
        m = re.search(r"\(([^)]+)\)", term)
        if m:
            abbr = m.group(1).strip()
            if len(abbr) <= 6:
                res.append(abbr)
        return list(dict.fromkeys(res))

    def expand(self, term: str) -> List[str]:
        if not term:
            return []
        res = []
        mesh_suggestions: List[str] = []
        try:
            if self.has_entrez:
                # Try MeSH lookup; Entrez can search 'mesh' or use term to find headings
                try:
                    handle = self.Entrez.esearch(db="mesh", term=term, retmax=5)
                    rec = self.Entrez.read(handle)
                    ids = rec.get("IdList", [])
                    for mid in ids:
                        try:
                            handle2 = self.Entrez.efetch(db="mesh", id=mid, retmode="xml")
                            txt = handle2.read()
                            # crude extraction of DescriptorName
                            m = re.search(r"<DescriptorName[^>]*>([^<]+)</DescriptorName>", txt)
                            if m:
                                mesh_suggestions.append(m.group(1))
                        except Exception:
                            continue
                except Exception:
                    logger.debug("Entrez MeSH lookup failed for '%s'", term)

            # Always include simple synonyms
            synonyms = self._simple_synonyms(term)
            res.extend(synonyms)

        except Exception as e:
            logger.debug("MeshExpander.expand error: %s", e)

        # Deduplicate and preserve order
        seen = set()
        out = []
        # preserve mesh suggestions first, then synonyms, then any other results
        for r in mesh_suggestions + res:
            if r and r not in seen:
                seen.add(r)
                out.append(r)

        # Log debug artifact with original, mesh suggestions, and synonyms for manual inspection
        try:
            debug = get_debug_manager()
            if debug.is_enabled():
                # safe filename
                safe = re.sub(r"[^0-9A-Za-z._-]", "_", term).strip("_")[:120]
                debug.save_json(f"retrieval/mesh/{safe}.json", {
                    "original": term,
                    "mesh_suggestions": mesh_suggestions,
                    "synonyms": synonyms,
                    "expanded": out,
                })
        except Exception:
            logger.debug("Failed to write mesh debug artifact for %s", term)
        return out
