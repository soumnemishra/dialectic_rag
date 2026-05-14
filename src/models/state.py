from typing import List, Annotated, Dict, Any, Optional
import operator
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from .schemas import PICO, EvidenceItem, EpistemicResult
from .enums import EpistemicState

def deduplicate_evidence(x: List[EvidenceItem], y: List[EvidenceItem]) -> List[EvidenceItem]:
    """Reducer that appends lists and deduplicates by PMID to support iterative agent loops."""
    combined = x + y
    seen_pmids = set()
    unique = []
    for item in combined:
        # Handle both EvidenceItem objects and dictionaries
        pmid = str(getattr(item, "pmid", "") or (item.get("pmid", "") if isinstance(item, dict) else ""))
        if pmid not in seen_pmids:
            unique.append(item)
            seen_pmids.add(pmid)
    return unique

class GraphState(TypedDict):
    """
    Master shared state for the Publishable Epistemic RAG Pipeline.
    """
    # -- Core query & Intent --
    original_question: str
    mcq_options: Optional[str]
    intent: Optional[str]
    risk_level: Optional[str]
    pico: Optional[PICO]
    
    # -- Evidence Pool --
    # Uses deduplication reducer to enable multi-step retrieval (Hardening v2)
    evidence_pool: Annotated[List[EvidenceItem], deduplicate_evidence]
    
    # -- Contrastive Retrieval --
    retrieved_docs: Dict[str, List[Dict]]
    step_notes: List[str]
    claim_clusters: List[List[EvidenceItem]]
    
    # -- Epistemic Analysis --
    temporal_result: Optional[EpistemicResult]
    consensus_state: Optional[EpistemicState]
    epistemic_result: Optional[EpistemicResult]
    abstention_rationale: Optional[str]
    
    # -- Synthesis & Final Answer --
    candidate_answers: List[str]
    candidate_answer: str
    final_reasoning: str
    abstention_triggered: bool
    
    # -- Traceability & Safety --
    extracted_claims: List[Dict[str, Any]]
    candidate_stances: Dict[str, Any]
    fused_beliefs: Dict[str, Any]
    temporal_shift: Dict[str, Any]
    epistemic_state: Optional[str]
    safety_flags: Annotated[List[str], operator.add]
    trace_events: Annotated[List[Dict[str, Any]], operator.add]
    trace_id: Optional[str]
    trace_created_at: Optional[str]
