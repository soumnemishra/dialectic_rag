
# state.py - Data flow for the clinical epistemic RAG pipeline
'''
Every module (intent, retrieval, scoring, synthesis) reads from this shared state,
performs its specific reasoning task, and writes findings back into the state.
'''

from typing import List, Annotated, Optional, TypedDict, Literal, Dict, Any
import operator
from pydantic import BaseModel, Field
import logging

# =============================================================================
# Pydantic Models — LLM output validation
# =============================================================================

# validates output of the single question-answering task.
class QAAnswerFormat(BaseModel):
    clinical_reasoning: str = Field(description="Chain-of-thought reasoning about the question and context")
    predicted_letter: Literal["A", "B", "C", "D", "UNKNOWN"] = Field(
        description="The final multiple-choice selection"
    )
    epistemic_confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score from 0.0 to 1.0",
    )

########--------------------- Clinical Intent Module Format ---------------------------------------------------#######
class ClinicalIntentFormat(BaseModel):
    intent:              str   = Field(description="informational | diagnostic | therapeutic | mechanism")
    risk_level:          str   = Field(description="low | medium | high")
    requires_disclaimer: bool  = Field(description="Mandatory for therapeutic/diagnostic queries")
    needs_guidelines:    bool  = Field(description="True if guidelines are required")
    confidence:          float = Field(ge=0.0, le=1.0, description="Classification certainty 0.0–1.0")
    reasoning:           str   = Field(description="One sentence explanation of the classification")

#------------------- Evidence Polarity Module Format ----------------------------------#######
class EvidencePolarity(BaseModel):
    """
    Quality of the evidence: support, refute, or insufficient.
    """
    polarity:   str   = Field(default="insufficient",
                              description="strong_support | weak_support | refute | insufficient")
    confidence: float = Field(default=0.0, description="0.0–1.0")
    reasoning:  str   = Field(default="", description="One sentence explanation")

    def to_dict(self) -> dict:
        return {"polarity": self.polarity, "confidence": self.confidence, "reasoning": self.reasoning}


# =============================================================================
# State TypedDicts — shared memory between modules
# =============================================================================

class QAAnswerState(TypedDict):
    clinical_reasoning: str
    predicted_letter: str
    epistemic_confidence: float

class ClinicalIntentState(TypedDict):
    intent:              str
    risk_level:          str
    requires_disclaimer: bool
    needs_guidelines:    bool
    confidence:          float
    reasoning:           str

class RagState(TypedDict):
    """Temporary state for a single RAG execution."""
    question:         str
    documents:        List[str]
    doc_ids:          List[str]
    notes:            List[str]
    final_raw_answer: QAAnswerState
    intent:           str
    risk_level:       str
    safety_flags:     List[str]
    evidence_polarity: Dict[str, Any]


class GraphState(TypedDict):
    """
    Master shared state for the Epistemic Reasoning Pipeline.

    Fields are grouped by which module writes them:
      ClinicalIntentModule  → intent, risk_level, requires_disclaimer,
                              needs_guidelines, confidence, reasoning, safety_flags
      RagDirectModule       → final_answer, step_output, step_docs_ids, step_notes
      EvidencePolarityModule → evidence_polarity
      EvidenceGovernance    → governance_decision
      DecisionAlignment     → final_answer (may modify), safety_flags
      SafetyCritic          → final_answer (may refine), safety_flags
    """
    # ── Core query ──────────────────────────────────────────────────
    original_question: str
    mcq_options: str
    chat_history: List[Dict[str, str]]

    # ── Execution history ────────────────────────────────────────────
    final_answer:     str
    final_answer_raw: str
    predicted_letter: str
    candidate_answer: str
    switch_applied: bool

    # ── Current execution step data (retrieved context) ──────────────
    step_output:      Annotated[List[Dict[str, Any]], operator.add]
    step_docs_ids:    Annotated[List[Any], operator.add]
    step_notes:       Annotated[List[str], operator.add]

    # ── Clinical intent classification ───────────────────────────────
    intent:              str
    risk_level:          str
    requires_disclaimer: bool
    needs_guidelines:    bool
    confidence:          float
    reasoning:           str

    # ── Evidence quality pipeline ────────────────────────────────────
    evidence_polarity:  Annotated[Dict[str, Any], lambda _old, new: new]
    governance_decision: Literal["accept", "abstain"]
    opposing_evidence_found: bool
    dialectical_metadata: Dict[str, Any]

    # ── Safety / audit ───────────────────────────────────────────────
    safety_flags:       Annotated[List[str], operator.add]

    # ── Temporal Belief Revision (TBR) ───────────────────────────────
    tcs_score:          float
    temporal_conflicts: Annotated[List[Dict[str, Any]], operator.add]
    overturned_pmids:   Annotated[List[str], operator.add]

    # ── Cohort Applicability / External Validity ─────────────────────
    applicability_score: float

    # ── Reproducibility Potential Score (RPS) ────────────────────────
    rps_scores:         Annotated[List[Dict[str, Any]], operator.add]

    # ── Contrastive Dialectical Synthesis (CDS) ──────────────────────
    thesis_docs:        Annotated[List[str], operator.add]
    antithesis_docs:    Annotated[List[str], operator.add]
    dialectic_synthesis: Annotated[Dict[str, Any], lambda _old, new: new]
    dialectical_retrieval_done: bool
    controversy_label:  str

    # ── Epistemic Uncertainty Propagation (EUP) ──────────────────────
    eus_per_claim:      Annotated[Dict[str, float], lambda _old, new: new]
    belief_intervals:   Annotated[Dict[str, Any], lambda _old, new: new]
    eus_override:       float

    # ── Evaluation ───────────────────────────────────────────────────
    evaluation_mode:    bool

    # ── Epistemic traceability ───────────────────────────────────────
    trace_id: Annotated[str, lambda _old, new: new]
    trace_created_at: Annotated[str, lambda _old, new: new]
    trace_events: Annotated[List[Dict[str, Any]], operator.add]

    # ── New DIALECTIC-RAG Pipeline Fields ────────────────────────────
    pico: Dict[str, Any]
    candidate_answers: List[str]
    retrieved_docs: Dict[str, Any]
    evidence_pool: List[Any]
    extracted_claims: List[Dict[str, Any]]
    claim_clusters: List[Dict[str, Any]]
    candidate_stances: Dict[str, Any]
    temporal_shift: Dict[str, Any]
    fused_beliefs: Dict[str, Any]
    epistemic_state: str
    abstention_triggered: bool



#############################################################