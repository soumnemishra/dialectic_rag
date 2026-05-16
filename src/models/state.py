from typing import List, Annotated, Dict, Any, Optional
import operator
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from .schemas import PICO, EvidenceItem, EpistemicResult
from .enums import EpistemicState

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
    # Explicitly enforce replacement of the list to avoid concatenation bugs
    evidence_pool: Annotated[List[EvidenceItem], lambda x, y: y]
    
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
