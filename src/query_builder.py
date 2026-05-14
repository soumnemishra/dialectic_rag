# FILE: src/query_builder.py
"""
PICO-lite PubMed Query Builder.

Converts free-text medical questions into high-recall, high-precision PubMed queries
using PICO decomposition, term normalization, MeSH injection, and proper Boolean assembly.

Example Usage:
    from src.query_builder import PubMedQueryBuilder, PICOQuery
    
    builder = PubMedQueryBuilder()
    
    # From structured input
    pico = PICOQuery(
        population=["melanoma", "cutaneous melanoma"],
        intervention=["treatment", "therapy"],
        modifiers=["stage II", "stage 2"]
    )
    queries = builder.build_query(pico)
    
    # From free text (requires LLM decomposition)
    queries = await builder.from_free_text("What is the latest treatment for stage II melanoma?")
"""
#Convert a messy human medical question into a precise, high-recall PubMed Boolean query.
import logging
import os
import re
#from dataclasses import dataclass, field
#from enum import Enum
from typing import List, Optional, Dict, Tuple, Any

from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)


# =============================================================================
# PICO Query Model
# =============================================================================
#  BASE MODEL A strict gatekeeper that checks data before letting it in.
# so every pico query is checked and validated before it is used.
class PICOQuery(BaseModel): #BASE MODEL IS THE DATA VALIDATION ENGINE
    """
    PICO-lite decomposition of a medical query.
    
    Attributes:
        population: Disease/condition terms (P in PICO).
        intervention: Treatment/intervention/topic terms (I in PICO).
        comparison: Comparator terms (C in PICO) - optional.
        outcome: Outcome terms (O in PICO) - optional.
        modifiers: Additional qualifiers (stage, age, etc.).
        study_types: Optional study type filters.
        suggested_mesh_terms: Suggested MeSH terms for ontology expansion.
        date_range: Optional publication date range (start_year, end_year).
        humans_only: Whether to filter for human studies.
    """
    # the field helper function of the pydantic always reconfigure the list  for each query 
    population: List[str] = Field(  # pydantic helper function to configure the field 
        default_factory=list,
        description="Disease/condition/population terms"   #  population=123(wrong) must be some disease name
    )
    intent: Optional[str] = Field(
        default=None,
        description="Clinical intent (treatment, diagnosis, etc.)"
    )
    intervention: List[str] = Field(
        default_factory=list,
        description="Intervention/treatment/topic terms"   # the description contains additional isntruction to the llm 
    )
    comparison: List[str] = Field(
        default_factory=list,
        description="Comparator terms (optional)"
    )
    outcome: List[str] = Field(
        default_factory=list,
        description="Outcome terms (optional)"
    )
    modifiers: List[str] = Field(
        default_factory=list,
        description="Modifier terms (stage, severity, etc.)"
    )
    study_types: List[str] = Field(
        default_factory=list,
        description="Study type filters (Clinical Trial, RCT, etc.)"
    )
    suggested_mesh_terms: List[str] = Field(
        default_factory=list,
        description="Suggested exact PubMed MeSH terms relevant to the population and intervention."
    )
    date_range: Optional[Tuple[int, int]] = Field(
        default=None,
        description="Publication date range (start_year, end_year)"
    )
    humans_only: bool = Field(
        default=True,
        description="Filter for human studies"
    )
    differential_candidates: List[str] = Field(
        default_factory=list,
        description="Candidate diseases for differential diagnosis"
    )

#################################################################################################################################
# =============================================================================
# Normalization Dictionaries (Controlled Vocabulary)
# =============================================================================
'''
====>These exist to translate intent → PubMed

=====>They prevent missing key papers

======>They embed domain knowledge

=======>They make RAG reliable

=======>Without them, answers degrade badly'''
# Common topic/intervention normalizations
#just like if user say treatment it will be normalized to therapy, management, Therapeutics[MeSH]
# and MESH IS MEDICAL SUBJECT HEADING  ( THE MAIN INDEXING CRITERIA OF THE PUBMED )

#“Search for the word melanoma only in the Title and Abstract of papers. OR WHERE THE MATCH HAPPENS THIS INCREASES THE RECALL ”
TOPIC_NORMALIZATIONS: Dict[str, List[str]] = {
    "treatment": ["treatment", "therapy", "management", "Therapeutics[MeSH]"],
    "diagnosis": ["diagnosis", "diagnostic", "screening", "detection", "Diagnosis[MeSH]"],
    "prognosis": ["prognosis", "survival", "outcome", "mortality", "Prognosis[MeSH]"],
    "prevention": ["prevention", "prophylaxis", "preventive", "Prevention and Control[MeSH]"],
    "etiology": ["etiology", "cause", "pathogenesis", "risk factors", "Etiology[MeSH]"],
    "epidemiology": ["epidemiology", "prevalence", "incidence", "Epidemiology[MeSH]"],
    "side effects": ["side effects", "adverse effects", "toxicity", "complications"],
    "guidelines": ["guidelines", "recommendations", "consensus", "Practice Guidelines[pt]"],
}

#  COMMON STUDY TYPE NORMALIZATIONS AND PUBLICATION TYPE NORMALIZATIONS  #
# Study type publication type tags
STUDY_TYPE_TAGS: Dict[str, str] = {
    "clinical trial": "Clinical Trial[pt]",
    "rct": "Randomized Controlled Trial[pt]",
    "randomized controlled trial": "Randomized Controlled Trial[pt]",
    "meta-analysis": "Meta-Analysis[pt]",
    "systematic review": "Systematic Review[pt]",
    "case report": "Case Reports[pt]",
    "review": "Review[pt]",
    "guideline": "Practice Guideline[pt]",
}



# =============================================================================
# Intent-Aware Structures
# =============================================================================

class IntentContext(BaseModel):
    """
    Context information for clinical intent classification.
    
    Attributes:
        intent: Clinical intent category (treatment, diagnosis, etc.)
        risk_level: Risk level (high, medium, low).
        requires_disclaimer: Whether a clinical disclaimer is needed.
        needs_guidelines: Whether clinical guidelines should be retrieved.
        confidence: Confidence score (0.0-1.0) of classification.
        reasoning: Brief explanation of the classification.
    """
    intent: str = Field(
        description="Intent category: treatment|diagnosis|prognosis|etiology|mechanism|differential_diagnosis|adverse_effects|guidelines|epidemiology"
    )
    risk_level: str = Field(
        default="medium",
        description="Risk level: high|medium|low"
    )
    requires_disclaimer: bool = Field(
        default=True,
        description="Whether clinical disclaimer required"
    )
    needs_guidelines: bool = Field(
        default=False,
        description="Whether clinical guidelines should be prioritized"
    )
    confidence: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Confidence in classification (0.0-1.0)"
    )
    reasoning: str = Field(
        default="",
        description="Brief explanation of intent classification"
    )


class RetrievalDiagnostics(BaseModel):
    """
    Structured diagnostic record for a retrieval attempt.
    
    Attributes:
        original_question: The user's question as submitted.
        intent: Classified clinical intent.
        pico: PICO decomposition result.
        query_ladder: List of queries (strict, moderate, broad).
        selected_query: The query actually used for retrieval.
        hit_counts: Hit counts for each query in ladder.
        retry_steps: Log of retry attempts and their outcomes.
        final_pmids: PMIDs successfully retrieved.
        timestamp: When retrieval occurred.
    """
    original_question: str = Field(description="Original user question")
    intent: str = Field(
        default="unknown",
        description="Classified intent"
    )
    pico: Optional[Dict[str, Any]] = Field(
        default=None,
        description="PICO decomposition as dict"
    )
    query_ladder: List[str] = Field(
        default_factory=list,
        description="[strict_query, moderate_query, broad_query]"
    )
    selected_query: Optional[str] = Field(
        default=None,
        description="Query actually used"
    )
    hit_counts: List[int] = Field(
        default_factory=list,
        description="Hit count for each query in ladder"
    )
    retry_steps: List[str] = Field(
        default_factory=list,
        description="Log of retry decisions and reasons"
    )
    final_pmids: List[str] = Field(
        default_factory=list,
        description="PMIDs successfully retrieved"
    )
    timestamp: str = Field(
        default_factory=lambda: __import__("datetime").datetime.utcnow().isoformat(),
        description="ISO timestamp of retrieval"
    )


# =============================================================================
# Intent-Specific Query Strategies
# =============================================================================

class IntentQueryStrategy:
    """
    Maps clinical intent to PICO priorities and query templates.
    
    Ensures queries are tailored to the type of evidence being sought.
    """
    
    # Intent → (population_weight, intervention_weight, outcome_weight, study_type_weight)
    INTENT_EMPHASIS: Dict[str, Tuple[float, float, float, float]] = {
        "treatment": (0.7, 1.0, 0.9, 0.8),           # Emphasis: intervention and outcome
        "diagnosis": (1.0, 0.9, 0.8, 0.5),           # Emphasis: population and test characteristics
        "prognosis": (1.0, 0.5, 1.0, 0.9),           # Emphasis: population and outcome
        "etiology": (0.9, 0.8, 1.0, 0.6),            # Emphasis: outcome (risk factors)
        "mechanism": (0.6, 1.0, 0.8, 0.3),           # Emphasis: intervention and mechanism
        "differential_diagnosis": (1.0, 0.9, 0.7, 0.4),  # Emphasis: population
        "adverse_effects": (0.8, 1.0, 1.0, 0.7),     # Emphasis: intervention and outcome
        "guidelines": (0.8, 0.8, 0.8, 1.0),          # Emphasis: study type (guidelines)
        "epidemiology": (1.0, 0.5, 0.6, 0.7),        # Emphasis: population and outcomes
    }
    
    # Intent → suggested study types
    INTENT_STUDY_TYPES: Dict[str, List[str]] = {
        "treatment": ["RCT", "Clinical Trial", "Meta-Analysis", "Systematic Review"],
        "diagnosis": ["Systematic Review", "Meta-Analysis", "Clinical Trial"],
        "prognosis": ["Cohort", "Meta-Analysis", "Systematic Review"],
        "etiology": ["Cohort", "Case-Control", "Meta-Analysis"],
        "mechanism": ["Review", "Research"],  # Broader scope for mechanisms
        "differential_diagnosis": ["Case Report", "Review", "Clinical Trial"],
        "adverse_effects": ["RCT", "Observational Study", "Systematic Review"],
        "guidelines": ["Practice Guideline"],
        "epidemiology": ["Observational Study", "Epidemiologic Studies"],
    }
    
    @staticmethod
    def get_adversarial_suffixes(intent: str) -> List[str]:
        """
        Get intent-specific adversarial query suffixes for dialectical retrieval.
        
        Returns list of suffixes that retrieve contradictory evidence.
        """
        strategies = {
            "treatment": [
                "no benefit",
                "adverse effects",
                "inferior to placebo",
                "complications",
                "ineffective",
            ],
            "diagnosis": [
                "false positive",
                "low sensitivity",
                "low specificity",
                "misdiagnosis",
            ],
            "mechanism": [
                "no association",
                "confounding factors",
                "opposite effect",
                "null finding",
            ],
            "differential_diagnosis": [
                "benign causes",
                "alternative diagnosis",
                "non-malignant",
                "mimics",
            ],
            "prognosis": [
                "poor prognosis",
                "negative outcome",
                "mortality",
                "complications",
            ],
            "adverse_effects": [
                "safe",
                "well tolerated",
                "no adverse effects",
                "prevention",
            ],
            "epidemiology": [
                "low prevalence",
                "rare",
                "uncommon",
                "declining incidence",
            ],
            "etiology": [
                "protective",
                "preventive",
                "no association",
            ],
        }
        return strategies.get(intent, ["however", "contradicts", "contrary to"])


# =============================================================================
# Query Builder
# =============================================================================

class PubMedQueryBuilder:
    """
    Builds high-recall, high-precision PubMed queries from PICO components.
    
    Uses:
    - MeSH terms with free-text fallback
    - Proper Boolean assembly (OR within concepts, AND between)
    - Title/Abstract field tags for precision
    - Optional precision boosters (study type, date, species)
    """
    
    def __init__(
        self,
        use_mesh: bool = True,  # Whether to include MeSH terms. AND SOME TIME WE NEED FREE TEXT OR DEBUG BEHAVIOUR USING THINGS CAN HELP OUT 
        use_tiab: bool = True,  # Whether to include title/abstract field tags.
        default_humans_filter: bool = True,  #ADDED TO HUMAN SPECIFIC FILTERS 
    ):
        """
        Initialize query builder.
        
        Args:
            use_mesh: Include MeSH terms in queries.
            use_tiab: Include title/abstract field tags.
            default_humans_filter: Add Humans[MeSH] filter by default.
        """
        self.use_mesh = use_mesh  #REFFERENCE THE CURRENT OBJECT 
        self.use_tiab = use_tiab
        self.default_humans_filter = default_humans_filter
        self.max_query_chars = int(os.getenv("MRAGE_QUERY_MAX_CHARS", "260"))
        self.max_query_concepts = int(os.getenv("MRAGE_QUERY_MAX_CONCEPTS", "3"))
        #EASY LOGGING AND DEBUGGING 
        logger.info(
            "PubMedQueryBuilder initialized",
            extra={"use_mesh": use_mesh, "use_tiab": use_tiab}
        )
    # BUILDS UP THE PUBMED QUERY FROM THE PICO COMPONENTS
    #PUBLIC FUNCTION THAT ACTUALLY CALLED 
    # THIS IS THE MAIN FUCNTION OF HOW THE QUERY IS BUILT UP 
    
    def build_query_with_intent(self, pico: "PICOQuery", intent: str) -> List[str]:
        """
        Build intent-aware query ladder from PICO components.
        
        Applies intent-specific emphasis to PICO components.
        For example:
        - treatment: emphasizes intervention + outcome
        - diagnosis: emphasizes population + test characteristics
        - mechanism: emphasizes intervention mechanisms
        
        Args:
            pico: PICO query decomposition.
            intent: Clinical intent (treatment, diagnosis, prognosis, etc.)
        
        Returns:
            Ordered list of PubMed query strings from strictest to broadest.
        """
        # For now, delegate to standard build_query
        # Intent-based weighting could be added here to modify PICO emphasis
        queries = self.build_query(pico)
        
        # Log the intent-guided building
        logger.info(
            "Query ladder built with intent guidance",
            extra={
                "intent": intent,
                "query_count": len(queries),
                "pico_population": len(pico.population),
                "pico_intervention": len(pico.intervention),
            }
        )
        
        return queries
    
    def build_query(self, pico: "PICOQuery") -> List[str]:
        """
        Build a fallback ladder of PubMed queries from PICO components.
        
        Args:
            pico: PICO query decomposition.
        
        Returns:
            Ordered list of PubMed query strings from strictest to broadest.
        
        Example:
            >>> pico = PICOQuery(
            ...     population=["melanoma"],
            ...     intervention=["treatment"],
            ...     modifiers=["stage II"]
            ... )
            >>> builder.build_query(pico)
            ['((melanoma[tiab])) AND ...', '((melanoma[tiab])) AND ...', '((melanoma[tiab])) AND ...']
        """
        use_humans_filter = pico.humans_only or self.default_humans_filter

        population_block = self._build_population_block(
            population_terms=pico.population,
            suggested_mesh_terms=pico.suggested_mesh_terms,
        )
        intervention_block = self._build_concept_block(
            terms=pico.intervention,
            normalization_map=TOPIC_NORMALIZATIONS,
            concept_name="intervention",
            population_terms=pico.population
        )
        outcome_block = self._build_concept_block(
            terms=pico.outcome,
            concept_name="outcome",
            population_terms=pico.population
        )
        modifier_block = self._build_modifier_block(pico.modifiers) if pico.modifiers else ""

        # Special handling for differential diagnosis: add candidates to population
        if pico.differential_candidates:
            candidate_block = self._build_concept_block(
                terms=pico.differential_candidates,
                concept_name="population"
            )
            if candidate_block:
                if population_block:
                    population_block = f"({population_block} OR {candidate_block})"
                else:
                    population_block = candidate_block

        strict_query = self._assemble_query(
            [population_block, intervention_block, outcome_block, modifier_block],
            boosters=self._build_boosters(
                study_types=pico.study_types,
                date_range=pico.date_range,
                include_date=True,
                include_humans=use_humans_filter,
            ),
        )
        moderate_query = self._assemble_query(
            [population_block, intervention_block],
            boosters=self._build_boosters(
                study_types=pico.study_types,
                date_range=None,
                include_date=False,
                include_humans=use_humans_filter,
            ),
        )
        broad_query = self._assemble_query(
            [population_block, intervention_block],
            boosters=[],
        )

        disease_hint = pico.population[0].strip() if pico.population else None
        if not disease_hint and pico.suggested_mesh_terms:
            disease_hint = pico.suggested_mesh_terms[0].strip()
        queries = [
            self.enforce_query_limits(strict_query, disease_hint=disease_hint, intent=getattr(pico, "intent", None)),
            self.enforce_query_limits(moderate_query, disease_hint=disease_hint, intent=getattr(pico, "intent", None)),
            self.enforce_query_limits(broad_query, disease_hint=disease_hint, intent=getattr(pico, "intent", None)),
        ]
        
        queries = self._enforce_entity_in_ladder(queries, disease_hint)

        logger.info(
            "Query ladder built",
            extra={
                "population_count": len(pico.population),
                "intervention_count": len(pico.intervention),
                "mesh_count": len(pico.suggested_mesh_terms),
                "query_count": len(queries),
                "strict_query_length": len(strict_query),
                "moderate_query_length": len(moderate_query),
                "broad_query_length": len(broad_query),
            }
        )

        return queries

    def _strip_wrapping_parens(self, text: str) -> str:
        cleaned = (text or "").strip()
        if not (cleaned.startswith("(") and cleaned.endswith(")")):
            return cleaned
        depth = 0
        for idx, ch in enumerate(cleaned):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0 and idx != len(cleaned) - 1:
                    return cleaned
        return cleaned[1:-1].strip()

    def _split_top_level_and(self, query: str) -> List[str]:
        parts: List[str] = []
        if not query:
            return parts
        depth = 0
        start = 0
        upper = query.upper()
        i = 0
        while i < len(query):
            ch = query[i]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth = max(0, depth - 1)
            if depth == 0 and upper[i:i + 5] == " AND ":
                parts.append(query[start:i].strip())
                start = i + 5
                i += 4
            i += 1
        tail = query[start:].strip()
        if tail:
            parts.append(tail)
        return [p for p in parts if p]

    def _format_term_block(self, term: str) -> str:
        cleaned = (term or "").strip()
        if not cleaned:
            return ""
        if re.search(r"\[[A-Za-z]+\]", cleaned):
            return cleaned
        return self._format_tiab(cleaned)

    def _pick_primary_term(self, concept_block: str) -> str:
        block = self._strip_wrapping_parens(concept_block)
        parts = re.split(r"\s+OR\s+", block, flags=re.IGNORECASE)
        return parts[0].strip() if parts else block

    def _default_secondary_block(self) -> str:
        return f"{self._format_tiab('clinical features')} OR {self._format_tiab('diagnosis')}"

    def _normalize_entity(self, entity: str) -> str:
        cleaned = (entity or "").strip()
        if not cleaned:
            return ""
        cleaned = re.sub(r"\[[A-Za-z]+\]", "", cleaned)
        cleaned = cleaned.strip('"').strip("'")
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.lower().strip()

    def _query_contains_entity(self, query: str, entity: str) -> bool:
        normalized = self._normalize_entity(entity)
        if not normalized:
            return True
        haystack = (query or "").lower()
        if not haystack:
            return False
        if " " in normalized:
            return normalized in haystack
        return re.search(rf"\b{re.escape(normalized)}\b", haystack) is not None

    def ensure_entity_in_query(self, query: str, entity: Optional[str]) -> str:
        cleaned = (query or "").strip()
        if not entity:
            return cleaned
        if self._query_contains_entity(cleaned, entity):
            return cleaned
        return f"({self._format_term_block(entity)})"

    def _enforce_entity_in_ladder(self, queries: List[str], entity: Optional[str]) -> List[str]:
        if not entity:
            return queries
        
        valid_queries = []
        for candidate in queries:
            if self._query_contains_entity(candidate, entity):
                valid_queries.append(candidate)
        
        if not valid_queries:
            logger.warning(
                "Query ladder dropped core entity; returning disease-only fallback ladder",
                extra={"entity": entity},
            )
            # Create a robust fallback ladder
            return [
                f"({self._format_term_block(entity)})",
                f"({self._format_term_block(entity)}) AND Humans[MeSH]",
                f'"{self._normalize_entity(entity)}"[tiab]'
            ]
            
        return valid_queries

    def _is_clinically_complex(self, intent: Optional[str]) -> bool:
        """Check if the intent warrants preserving query complexity."""
        if not intent:
            return False
        return intent in {
            "differential_diagnosis",
            "treatment",
            "diagnosis",
            "prognosis",
        }

    def _is_lab_token(self, term: str) -> bool:
        """Check if a term is a generic clinical descriptor/lab token."""
        _LAB_TOKENS = {
            "ast", "alt", "lft", "lfts", "electrolyte", "electrolytes", "ammonia",
            "eeg", "lab", "labs", "value", "values", "vital", "vitals", "sign", "signs",
            "clinical", "features", "finding", "findings", "deranged", "elevated"
        }
        q = str(term or "").strip().lower()
        q = re.sub(r"\[[a-z]+\]", "", q)
        q = q.strip('"').strip("'")
        return q in _LAB_TOKENS

    def enforce_query_limits(
        self,
        query: str,
        disease_hint: Optional[str] = None,
        primary_symptom: Optional[str] = None,
        max_concepts: Optional[int] = None,
        max_chars: Optional[int] = None,
        intent: Optional[str] = None,
    ) -> str:
        """Simplify overlong queries to prevent retrieval starvation."""
        cleaned = (query or "").strip()
        if not cleaned:
            return cleaned

        # Bypass truncation for clinically complex intents
        if self._is_clinically_complex(intent):
            logger.info(f"Bypassing query truncation for clinically complex intent: {intent}")
            return cleaned

        concept_limit = max_concepts if max_concepts is not None else self.max_query_concepts
        char_limit = max_chars if max_chars is not None else self.max_query_chars
        parts = self._split_top_level_and(cleaned)

        if len(cleaned) <= char_limit and len(parts) <= concept_limit:
            return cleaned

        # Extreme length truncation (> 500 chars)
        if len(cleaned) > 500:
            logger.info("Query exceeds extreme length limit (>500); applying deterministic reduction.")
            # Keep population and intervention, drop outcome and modifiers
            reduced_parts = parts[:2]
            reduced_query = " AND ".join(reduced_parts)
            if len(reduced_query) <= char_limit:
                logger.info(f"Extreme truncation reduced query to: {reduced_query}")
                return reduced_query

        disease_block = self._format_term_block(disease_hint) if disease_hint else self._strip_wrapping_parens(parts[0] if parts else cleaned)
        
        # If simplification reduces disease to a lab token, preserve original or abort.
        if self._is_lab_token(disease_block) and disease_hint:
             disease_block = self._format_term_block(disease_hint)

        if not disease_block:
            disease_block = self._format_term_block(cleaned) or cleaned
        
        symptom_block = ""
        if primary_symptom:
            symptom_block = self._format_term_block(primary_symptom)
        elif len(parts) > 1:
            symptom_block = self._format_term_block(self._pick_primary_term(parts[1]))

        if not symptom_block:
            symptom_block = self._default_secondary_block()

        # If disease_block is STILL a lab token after simplification attempts, return unsimplified prefix.
        if self._is_lab_token(disease_block):
            logger.warning("Simplification resulted in lab token; returning unsimplified prefix.")
            return cleaned[:max_chars] if max_chars else cleaned[:150]

        simplified = f"({disease_block}) AND ({symptom_block})"
        if disease_hint:
            simplified = self.ensure_entity_in_query(simplified, disease_hint)
        
        logger.info(
            "Query simplified | original='%s' | simplified='%s'",
            cleaned[:100],
            simplified[:100],
        )
        return simplified

    def _assemble_query(self, query_parts: List[str], boosters: Optional[List[str]] = None) -> str:
        parts = self._nonempty_parts(query_parts)
        booster_parts = self._nonempty_parts(boosters or [])

        base_query = " AND ".join(parts)
        if base_query and booster_parts:
            return f"({base_query}) AND ({' AND '.join(booster_parts)})"
        if booster_parts:
            return " AND ".join(booster_parts)
        return base_query

    def _nonempty_parts(self, parts: List[str]) -> List[str]:
        return [part.strip() for part in parts if isinstance(part, str) and part.strip()]

    def _build_boosters(
        self,
        study_types: List[str],
        date_range: Optional[Tuple[int, int]],
        include_date: bool,
        include_humans: bool,
    ) -> List[str]:
        boosters = []

        if study_types:
            study_filter = self._build_study_type_filter(study_types)
            if study_filter:
                boosters.append(study_filter)

        if include_date and date_range:
            boosters.append(self._build_date_filter(date_range))

        if include_humans:
            boosters.append("Humans[MeSH]")

        return boosters

    def _build_population_block(self, population_terms: List[str], suggested_mesh_terms: List[str]) -> str:
        """Build the population block plus dynamic MeSH expansion."""
        blocks = []

        population_block = self._build_concept_block(
            terms=population_terms,
            concept_name="population"
        )
        if population_block:
            blocks.append(population_block)

        if self.use_mesh and suggested_mesh_terms:
            mesh_block = self._build_mesh_block(suggested_mesh_terms)
            if mesh_block:
                blocks.append(mesh_block)

        blocks = self._nonempty_parts(blocks)
        if not blocks:
            return ""
        if len(blocks) == 1:
            return blocks[0]
        return f"({' OR '.join(blocks)})"

    def _build_mesh_block(self, terms: List[str]) -> str:
        """Format suggested MeSH terms as a PubMed OR block."""
        mesh_terms = []

        for term in terms:
            cleaned = term.strip()
            if not cleaned:
                continue
            if cleaned.lower().endswith("[mesh]"):
                mesh_terms.append(cleaned)
            else:
                mesh_terms.append(f"{cleaned}[MeSH]")

        unique_terms = self._nonempty_parts(list(dict.fromkeys(mesh_terms)))
        if not unique_terms:
            return ""

        return f"({' OR '.join(unique_terms)})"

     ###########################################HELPER FUCNTIONS #######################################################################   
    # THIS ARE THE HELPER FUNCTION THAT ARE CALLED BY THE BUILD_QUERY FUNCTION
    def _build_concept_block(
        self,
        terms: List[str],
        normalization_map: Optional[Dict[str, List[str]]] = None,
        concept_name: str = "concept",
        population_terms: Optional[List[str]] = None,
    ) -> str:
        """
        Build a concept block with OR-joined terms.
        
        Applies MeSH injection and term normalization.
        """
        expanded_terms = []
        
        for term in terms:
            term_lower = term.lower().strip()

            # Handle generic tokens by pairing with the primary population entity
            if concept_name in ("intervention", "outcome") and term_lower in _GENERIC_INTERVENTION:
                if population_terms and population_terms[0].strip():
                    primary_entity = population_terms[0].strip()
                    # Transform e.g., "effect" -> "dicloxacillin"[tiab] AND (effectiveness OR outcome OR effect)
                    paired_term = f'"{primary_entity}"[tiab] AND ({term_lower} OR outcome OR effectiveness)'
                    expanded_terms.append(paired_term)
                    logger.info(f"Paired generic term '{term_lower}' with entity '{primary_entity}'")
                elif concept_name == "intervention":
                    # If it's an intervention and we have no population, we MUST NOT drop it if it's the only one
                    # But if it's generic, it's useless alone. We'll keep it as tiab but log a warning.
                    expanded_terms.append(self._format_tiab(term))
                    logger.warning(f"Generic intervention '{term_lower}' kept without population context - high risk of noise")
                continue

            # Check for normalization expansion
            if normalization_map and term_lower in normalization_map:
                for normalized in normalization_map[term_lower]:
                    if "[MeSH]" in normalized or "[pt]" in normalized:
                        expanded_terms.append(normalized)
                    else:
                        expanded_terms.append(self._format_tiab(normalized))
            else:
                # Add as title/abstract term
                expanded_terms.append(self._format_tiab(term))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_terms = []
        for t in expanded_terms:
            if t.lower() not in seen:
                seen.add(t.lower())
                unique_terms.append(t)

        if not unique_terms:
            return ""
        
        # Join with OR
        block = " OR ".join(unique_terms)
        
        return f"({block})"
    
    def _build_modifier_block(self, modifiers: List[str]) -> str:
        """
        Build a modifier block with variant handling.
        
        Handles common variations like "stage II" vs "stage 2".
        """
        expanded_modifiers = []
        
        for mod in modifiers:
            mod_clean = mod.strip()
            
            # Add original
            expanded_modifiers.append(self._format_tiab(mod_clean))
            
            # Handle Roman numeral variants
            variants = self._get_numeral_variants(mod_clean)
            for variant in variants:
                expanded_modifiers.append(self._format_tiab(variant))
        
        # Remove duplicates
        unique_mods = list(dict.fromkeys(expanded_modifiers))
        
        block = " OR ".join(unique_mods)
        return f"({block})"
    
    def _format_tiab(self, term: str) -> str:
        """Format a term with title/abstract field tag."""
        # Quote multi-word terms
        if " " in term and not term.startswith('"'):
            term = f'"{term}"'
        
        if self.use_tiab:
            return f"{term}[tiab]"
        return term
    
    def _get_numeral_variants(self, text: str) -> List[str]:
        """
        Generate Roman/Arabic numeral variants.
        
        Examples:
            "stage II" -> ["stage 2"]
            "stage 2" -> ["stage II"]
            "type 1" -> ["type I"]
        """
        variants = []
        
        # Roman to Arabic mapping
        roman_to_arabic = {
            "I": "1", "II": "2", "III": "3", "IV": "4", "V": "5",
            "VI": "6", "VII": "7", "VIII": "8", "IX": "9", "X": "10"
        }
        arabic_to_roman = {v: k for k, v in roman_to_arabic.items()}
        
        # Check for Roman numerals (case insensitive)
        for roman, arabic in roman_to_arabic.items():
            pattern = rf'\b{roman}\b'
            if re.search(pattern, text, re.IGNORECASE):
                variant = re.sub(pattern, arabic, text, flags=re.IGNORECASE)
                if variant != text:
                    variants.append(variant)
        
        # Check for Arabic numerals
        for arabic, roman in arabic_to_roman.items():
            pattern = rf'\b{arabic}\b'
            if re.search(pattern, text):
                variant = re.sub(pattern, roman, text)
                if variant != text:
                    variants.append(variant)
        
        return variants
    
    def _build_study_type_filter(self, study_types: List[str]) -> str:
        """Build study type publication type filter."""
        type_tags = []
        
        for st in study_types:
            st_lower = st.lower().strip()
            if st_lower in STUDY_TYPE_TAGS:
                type_tags.append(STUDY_TYPE_TAGS[st_lower])
            else:
                # Try as-is with [pt] tag
                type_tags.append(f"{st}[pt]")
        
        if type_tags:
            return f"({' OR '.join(type_tags)})"
        return ""
    
    def _build_date_filter(self, date_range: Tuple[int, int]) -> str:
        """Build publication date filter."""
        start_year, end_year = date_range
        return f'("{start_year}"[dp] : "{end_year}"[dp])'
    
    def normalize_topic(self, topic: str) -> List[str]:
        """
        Expand a topic term using normalization dictionary.
        
        Args:
            topic: Raw topic term.
        
        Returns:
            List of normalized/expanded terms.
        """
        topic_lower = topic.lower().strip()
        
        if topic_lower in TOPIC_NORMALIZATIONS:
            return TOPIC_NORMALIZATIONS[topic_lower]
        
        return [topic]
    
    def get_mesh_term(self, disease: str) -> Optional[str]:
        """Deprecated compatibility helper retained for older call sites."""
        return None
    
    def select_query_by_count(
        self,
        query_counts: List[Tuple[str, int]],
        target_range: Tuple[int, int] = (5, 5000)
    ) -> Optional[str]:
        """
        Select the best query from a list of (query, count) tuples.
        
        Selection strategy:
        - Prefer counts in the target range (e.g., 5-5000)
        - If none in range, prefer the query with highest count > 0
        - Returns None if all counts are 0
        
        Args:
            query_counts: List of (query, hit_count) tuples
            target_range: Tuple of (min_count, max_count)
        
        Returns:
            Best query string or None if no acceptable query found
        """
        min_target, max_target = target_range
        
        # First pass: find queries in target range.
        in_range = [(q, c) for q, c in query_counts if c is not None and min_target <= c <= max_target]
        if in_range:
            # If the entire ladder is already within the target range, prefer
            # the strictest (first) query.
            if len(in_range) == len(query_counts):
                return in_range[0][0]

            # If the strictest query is too narrow (< min_target), then
            # expand to the first broader query that falls within range.
            first_count = query_counts[0][1]
            if first_count is not None and first_count < min_target:
                return in_range[0][0]

            # Otherwise prefer the query with the highest count among those
            # within the target range (more evidence, less likely to be noisy).
            best_in_range = max(in_range, key=lambda x: x[1])
            return best_in_range[0]
        
        # Second pass: prefer highest count > 0
        non_zero = [(q, c) for q, c in query_counts if c is not None and c > 0]
        if non_zero:
            best = max(non_zero, key=lambda x: x[1])
            return best[0]
        
        return None


# =============================================================================
# LLM-based PICO Decomposition Schema
# =============================================================================

class PICODecomposition(BaseModel):
    """
    Schema for LLM-based PICO decomposition of free-text queries.
    
    Used with Pydantic AI to structure LLM output.
    """
    population_terms: List[str] = Field(
        description='List of medical terms for the disease/pathogen. NEVER use standalone lab tokens like AST, ALT, ammonia. If a differential diagnosis is present, include the names of all candidate diseases.'
    )
    intervention_terms: List[str] = Field(
        description='Medical terms for the intervention/topic. Generate broad synonyms. Avoid overly specific vignette details.'
    )
    comparison_terms: List[str] = Field(
        default_factory=list,
        description='Medical terms for the comparator (e.g., ["placebo", "usual care"]).'
    )
    outcome_terms: List[str] = Field(
        default_factory=list,
        description='Medical terms for the outcome (e.g., ["mortality", "glycemic control"]).'
    )
    modifier_terms: List[str] = Field(
        default_factory=list,
        description='Medical terms for modifiers like stage, severity, or age group.'
    )
    suggested_study_types: List[str] = Field(
        default_factory=list,
        description="Suggested study types (e.g., 'RCT', 'systematic review')."
    )
    suggested_mesh_terms: List[str] = Field(
        default_factory=list,
        description='Predict 2-3 exact PubMed MeSH terms relevant to the population and intervention.'
    )
    differential_candidates: List[str] = Field(
        default_factory=list,
        description="If a differential diagnosis is requested, list the specific disease candidates being compared (e.g., ['Chikungunya', 'Dengue'])."
    )
    requires_recent: bool = Field(
        default=False,
        description="Whether the query implies need for recent/latest publications"
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize_payload(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        payload = dict(data)
        nested_pico = payload.get("pico")
        if isinstance(nested_pico, dict):
            payload.pop("pico", None)
            payload = {**payload, **nested_pico}

        alias_map = {
            "population": "population_terms",
            "intervention": "intervention_terms",
            "comparison": "comparison_terms",
            "outcome": "outcome_terms",
            "modifiers": "modifier_terms",
            "study_types": "suggested_study_types",
            "mesh_terms": "suggested_mesh_terms",
        }

        for source_key, target_key in alias_map.items():
            if source_key in payload and target_key not in payload:
                payload[target_key] = payload[source_key]

        list_fields = {
            "population_terms",
            "intervention_terms",
            "comparison_terms",
            "outcome_terms",
            "modifier_terms",
            "suggested_study_types",
            "suggested_mesh_terms",
            "differential_candidates",
        }

        for field_name in list_fields:
            value = payload.get(field_name)
            if isinstance(value, str):
                payload[field_name] = [value]
            elif value is None:
                payload[field_name] = []

        return payload
    
    def to_pico_query(self, recent_years: int = 5, intent: Optional[str] = None) -> PICOQuery:
        """Convert decomposition to PICOQuery."""
        import datetime
        
        date_range = None
        if self.requires_recent:
            current_year = datetime.datetime.now().year
            date_range = (current_year - recent_years, current_year)
        
        return PICOQuery(
            population=self.population_terms,
            intervention=self.intervention_terms,
            comparison=self.comparison_terms,
            outcome=self.outcome_terms,
            modifiers=self.modifier_terms,
            study_types=self.suggested_study_types,
            suggested_mesh_terms=self.suggested_mesh_terms,
            differential_candidates=self.differential_candidates,
            intent=intent,
            date_range=date_range,
            humans_only=True,
        )


def parse_markdown_json(raw_text: str) -> Dict[str, Any]:
    """Parse JSON from LLM responses that may include markdown or prose preambles."""
    import ast
    import json
    import re

    def _extract_complete_object(text: str) -> Optional[str]:
        starts = [idx for idx, ch in enumerate(text) if ch == "{"]
        for start in reversed(starts):
            candidate = text[start:]
            depth = 0
            end = -1
            in_str = False
            esc = False

            for i, ch in enumerate(candidate):
                if esc:
                    esc = False
                    continue
                if ch == "\\" and in_str:
                    esc = True
                    continue
                if ch == '"':
                    in_str = not in_str
                    continue
                if in_str:
                    continue

                if ch == "{":
                    depth += 1
                elif ch == "}":
                    if depth > 0:
                        depth -= 1
                        if depth == 0:
                            end = i
                            break

            if end != -1:
                return candidate[: end + 1]
        return None

    def _close_open_structures(text: str) -> str:
        stack: List[str] = []
        in_str = False
        esc = False
        out_chars: List[str] = []

        for ch in text:
            if esc:
                esc = False
                out_chars.append(ch)
                continue
            if ch == "\\" and in_str:
                esc = True
                out_chars.append(ch)
                continue
            if ch == '"':
                in_str = not in_str
                out_chars.append(ch)
                continue
            if in_str:
                out_chars.append(ch)
                continue

            if ch in "[{":
                stack.append(ch)
                out_chars.append(ch)
            elif ch == "}":
                if stack and stack[-1] == "{":
                    stack.pop()
                    out_chars.append(ch)
            elif ch == "]":
                if stack and stack[-1] == "[":
                    stack.pop()
                    out_chars.append(ch)
            else:
                out_chars.append(ch)

        if stack:
            out_chars.extend("}" if token == "{" else "]" for token in reversed(stack))
        return "".join(out_chars)

    def _repair_json_like(text: str) -> str:
        repaired = (text or "").strip()
        if not repaired:
            return repaired

        repaired = repaired.replace("\ufeff", "")
        repaired = repaired.replace("“", '"').replace("”", '"')
        repaired = repaired.replace("‘", "'").replace("’", "'")
        repaired = re.sub(r"^\s*json\s*", "", repaired, flags=re.IGNORECASE)

        first_brace = repaired.find("{")
        if first_brace != -1:
            repaired = repaired[first_brace:]

        repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
        repaired = re.sub(r'([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)', r'\1"\2"\3', repaired)

        complete = _extract_complete_object(repaired)
        if complete:
            repaired = complete
        else:
            repaired = _close_open_structures(repaired)

        repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
        return repaired.strip()

    def _try_loads(candidate: str) -> Optional[Dict[str, Any]]:
        if not candidate:
            return None

        try:
            obj = json.loads(candidate)
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            pass

        brace_match = re.search(r"\{[\s\S]*\}", candidate)
        if brace_match:
            brace_candidate = brace_match.group(0).strip()
            try:
                obj = json.loads(brace_candidate)
                return obj if isinstance(obj, dict) else None
            except json.JSONDecodeError:
                pass

        repaired = _repair_json_like(candidate)
        try:
            obj = json.loads(repaired)
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            pass

        py_like = re.sub(r"\btrue\b", "True", repaired, flags=re.IGNORECASE)
        py_like = re.sub(r"\bfalse\b", "False", py_like, flags=re.IGNORECASE)
        py_like = re.sub(r"\bnull\b", "None", py_like, flags=re.IGNORECASE)

        try:
            obj = ast.literal_eval(py_like)
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None

    text = (raw_text or "").strip()
    if not text:
        raise ValueError("Empty LLM response")

    tick = chr(96)
    fence = tick * 3

    candidates: List[str] = []

    json_fence_pattern = rf"{re.escape(fence)}json\s*([\s\S]*?){re.escape(fence)}"
    for match in re.finditer(json_fence_pattern, text, re.IGNORECASE):
        candidates.append(match.group(1).strip())

    any_fence_pattern = rf"{re.escape(fence)}\s*([\s\S]*?){re.escape(fence)}"
    for match in re.finditer(any_fence_pattern, text):
        candidates.append(match.group(1).strip())

    complete_obj = _extract_complete_object(text)
    if complete_obj:
        candidates.append(complete_obj)

    first_brace = text.find("{")
    if first_brace != -1:
        candidates.append(text[first_brace:])

    candidates.append(text)

    seen = set()
    for candidate in candidates:
        key = candidate.strip()
        if not key or key in seen:
            continue
        seen.add(key)

        parsed = _try_loads(candidate)
        if parsed is not None:
            return parsed

    raise ValueError(f"Failed to parse JSON from LLM: {text[:120]}...")


def parse_json_markdown(raw_text: str) -> Dict[str, Any]:
    """Alias for parse_markdown_json for compatibility with older call sites."""
    return parse_markdown_json(raw_text)


# =============================================================================
# Quick Helper Functions
# =============================================================================

def build_simple_query(
    disease: str,
    topic: str,
    modifiers: Optional[List[str]] = None,
    years: Optional[int] = None,
) -> List[str]:
    """
    Quick helper to build a simple PubMed query.
    
    Args:
        disease: Disease/condition term.
        topic: Topic/intervention term.
        modifiers: Optional modifier terms.
        years: Optional limit to recent N years.
    
    Returns:
        Ordered list of PubMed query strings from strictest to broadest.
    
    Example:
        #>>> build_simple_query("melanoma", "treatment", ["stage II"], years=5)
        ['((melanoma[tiab])) AND ...', '((melanoma[tiab])) AND ...', '((melanoma[tiab])) AND ...']
    """
    import datetime
    
    date_range = None
    if years:
        current_year = datetime.datetime.now().year
        date_range = (current_year - years, current_year)
    
    builder = PubMedQueryBuilder()
    expanded_intervention = builder.normalize_topic(topic)
    
    pico = PICOQuery(
        population=[disease],
        intervention=expanded_intervention[:2],  # use the first couple of expansions
        modifiers=modifiers or [],
        date_range=date_range,
    )
    
    queries = builder.build_query(pico)
    # Only return the strict, moderate, and broad queries (no extra fallback)
    return queries[:3]
