from enum import Enum

class EvidenceStance(str, Enum):
    SUPPORT = "SUPPORT"
    OPPOSE = "OPPOSE"
    NEUTRAL = "NEUTRAL"
    REFINE = "REFINE"

class StudyDesign(str, Enum):
    META_ANALYSIS = "meta_analysis"
    SYSTEMATIC_REVIEW = "systematic_review"
    RCT = "rct"
    COHORT = "cohort"
    CASE_CONTROL = "case_control"
    CASE_SERIES = "case_series"
    OTHER = "other"

class EpistemicState(str, Enum):
    SETTLED = "SETTLED"
    CONTESTED = "CONTESTED"
    EVOLVING = "EVOLVING"
    INSUFFICIENT = "INSUFFICIENT"
    FALSIFIED = "FALSIFIED"

class ResponseTier(str, Enum):
    FULL = "FULL"
    QUALIFIED = "QUALIFIED"
    ABSTAIN = "ABSTAIN"
