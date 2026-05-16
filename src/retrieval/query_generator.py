from typing import Dict, List, Optional
from dataclasses import dataclass
import logging
from src.utils.debug_utils import get_debug_manager
from src.retrieval.mesh_expander import MeshExpander

logger = logging.getLogger(__name__)


@dataclass
class PICO:
    population: Optional[str]
    intervention: Optional[str]
    comparator: Optional[str]
    outcome: Optional[str]
    intent: Optional[str]
    risk_level: Optional[str]


class QueryGenerator:
    """Generate supportive and contradictory PubMed queries for a hypothesis.

    Implements PICO-aware templating and simple MeSH/synonym expansion stubs.
    """

    CONTRADICTORY_CUES = [
        "no benefit",
        "ineffective",
        "failed trial",
        "contradictory",
        "versus placebo",
        "negative study",
        "did not improve",
        "associated with harm",
    ]

    def __init__(self, mesh_expander=None):
        # if no MeshExpander provided, create one (will attempt Entrez if available)
        self.mesh_expander = mesh_expander or MeshExpander()
        self.debug = get_debug_manager()

    def _expand_terms(self, text: str) -> List[str]:
        """Expand a phrase using MeSH/UMLS if available. Fallback: return original and a quoted form."""
        if not text:
            return []
        res = [text]
        res.append(f'"{text}"')
        # Call mesh_expander
        try:
            mesh_terms = self.mesh_expander.expand(text)
            for m in mesh_terms:
                if m not in res:
                    res.append(m)
        except Exception as e:
            logger.debug("MeSH expansion failed: %s", e)
        return res

    def _join_terms(self, terms: List[str]) -> str:
        """Join expanded terms into OR groups suitable for PubMed."""
        if not terms:
            return ""
        quoted = [t if ' ' not in t or t.startswith('"') else f'"{t}"' for t in terms]
        return "(" + " OR ".join(quoted) + ")"

    def generate(self, claim: str, pico: Optional[PICO], hypothesis: str) -> Dict[str, str]:
        """Return structured supportive and contradictory PubMed queries.

        - Build OR-groups per PICO element using MeSH and synonym expansion.
        - Use [Mesh] tag for population/disease primary term and [tiab] for text words.
        - AND across PICO concepts, OR within concept expansions.
        """
        groups = []

        def build_group(primary: str, extras: List[str], use_mesh: bool = False) -> str:
            items = []
            if use_mesh and primary:
                items.append(f'"{primary}"[Mesh]')
            # include primary and extras as [tiab]
            seen = set()
            for t in ([primary] if primary else []) + (extras or []):
                if not t:
                    continue
                key = t.strip()
                if key.lower() in seen:
                    continue
                seen.add(key.lower())
                # avoid adding the same string twice
                items.append(f'{key}[tiab]')
            if not items:
                return ""
            return "(" + " OR ".join(items) + ")"

        # Population / disease group
        pop = pico.population if pico and getattr(pico, "population", None) else None
        pop_extras = []
        if pop:
            try:
                pop_extras = self.mesh_expander.expand(pop)
            except Exception:
                pop_extras = []
        pop_group = build_group(pop, [e for e in pop_extras if e and e.lower() != (pop or "").lower()], use_mesh=True)
        if pop_group:
            groups.append(pop_group)

        # Intervention group
        intr = pico.intervention if pico and getattr(pico, "intervention", None) else None
        intr_extras = []
        if intr:
            try:
                intr_extras = self.mesh_expander.expand(intr)
            except Exception:
                intr_extras = []
        intr_group = build_group(intr, [e for e in intr_extras if e and e.lower() != (intr or "").lower()], use_mesh=False)
        if intr_group:
            groups.append(intr_group)

        # Outcome group
        outc = pico.outcome if pico and getattr(pico, "outcome", None) else None
        outc_extras = []
        if outc:
            try:
                outc_extras = self.mesh_expander.expand(outc)
            except Exception:
                outc_extras = []
        outc_group = build_group(outc, [e for e in outc_extras if e and e.lower() != (outc or "").lower()], use_mesh=False)
        if outc_group:
            groups.append(outc_group)

        # Additional modifiers (e.g., hospitalized, severe) - extract from claim heuristically
        modifiers = []
        for mod in ["hospitalized", "hospitalised", "severe", "critically ill"]:
            if mod in (claim or "").lower():
                modifiers.append(mod)
        if modifiers:
            mod_group = build_group(None, modifiers, use_mesh=False)
            if mod_group:
                groups.append(mod_group)

        # Final supportive query: AND across groups
        supportive_query = " AND ".join(groups) if groups else hypothesis

        # Contradictory query: add contradiction cue terms as OR group
        contradiction = " OR ".join([f'"{c}"[tiab]' for c in self.CONTRADICTORY_CUES])
        contradictory_query = f"({supportive_query}) AND ({contradiction})" if supportive_query else f"{hypothesis} AND ({contradiction})"

        # Save debug
        try:
            if self.debug.is_enabled():
                self.debug.save_json("retrieval/queries.json", {"supportive": supportive_query, "contradictory": contradictory_query})
        except Exception:
            pass

        return {"supportive": supportive_query, "contradictory": contradictory_query}
