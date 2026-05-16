from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from .enums import EvidenceStance, StudyDesign, EpistemicState, ResponseTier

class PICO(BaseModel):
    population: str
    intervention: str
    comparator: Optional[str] = "standard care"
    outcome: str
    risk_level: str = "unknown"

class StudyMetadata(BaseModel):
    sample_size: Optional[int] = None
    study_design: Optional[StudyDesign] = None
    has_p_value: bool = False
    has_CI: bool = False
    preregistration_id: Optional[str] = None  # NCT number
    year: Optional[int] = None
    source_type: Optional[str] = None  # "pubmed", "cochrane", etc.

class EvidenceItem(BaseModel):
    pmid: str
    title: str
    abstract: str
    claim: str  # extracted core claim
    metadata: StudyMetadata
    reproducibility_score: float
    applicability_score: float
    year: Optional[int] = None
    stance: Optional[EvidenceStance] = None
    nli_contradiction_prob: Optional[float] = None

class MassFunction(BaseModel):
    belief_true: float
    belief_false: float
    uncertainty: float  # mass on {true, false}
    conflict_K: Optional[float] = None

class EpistemicResult(BaseModel):
    state: EpistemicState
    belief: float
    uncertainty: float
    conflict: float
    temporal_shift_detected: bool
    response_tier: ResponseTier
    evidence_items: List[EvidenceItem]
    baseline_claim: Optional[str] = None
    current_belief: Optional[str] = None
    contradiction_events: List[Dict[str, Any]] = Field(default_factory=list)
