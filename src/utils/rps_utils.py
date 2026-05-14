"""RPS Utilities — Compatibility Layer for DIALECTIC-RAG.

This module provides legacy support for RPS scoring but redirects all 
mathematical computation to the canonical ReproducibilityScorer.

It handles robust extraction of study features from raw dictionaries or 
abstract text using regex patterns, then maps them to StudyMetadata for 
standardised scoring.

Standardized on the 4-feature formula:
    RPS = w_design · D + w_sample · S + w_pvalue · P + w_prereg · R
"""

import logging
import re
from typing import Any, Dict, Optional, Tuple

from src.models.schemas import StudyMetadata, StudyDesign
from src.models.enums import StudyDesign as StudyDesignEnum
from src.epistemic.reproducibility_scorer import ReproducibilityScorer
from src.config import epistemic_settings

logger = logging.getLogger(__name__)

# --- Compatibility Singleton ---
_SCORER = ReproducibilityScorer()

def _normalize_study_design(raw_design: str) -> StudyDesignEnum:
    """Map string study types to StudyDesign enum."""
    st = str(raw_design or "").strip().lower()
    if "meta-analysis" in st or "meta analysis" in st:
        return StudyDesignEnum.META_ANALYSIS
    if "systematic review" in st:
        return StudyDesignEnum.SYSTEMATIC_REVIEW
    if "randomized" in st or "randomised" in st or "rct" in st:
        return StudyDesignEnum.RCT
    if "cohort" in st:
        return StudyDesignEnum.COHORT
    if "case-control" in st or "case control" in st:
        return StudyDesignEnum.CASE_CONTROL
    if "case series" in st:
        return StudyDesignEnum.CASE_SERIES
    return StudyDesignEnum.OTHER

def _extract_sample_size(text: str) -> Optional[int]:
    """Lightweight regex parser for sample size."""
    if not text:
        return None
    s = text.replace(",", "")
    patterns = [
        r"\bN\s*=\s*([0-9]+)\b",
        r"\bn\s*=\s*([0-9]+)\b",
        r"([0-9]+)\s*(?:patients|participants|subjects|cases|individuals|women|men)",
        r"total of\s*([0-9]+)\s*",
        r"enrolled\s*([0-9]+)\s*",
    ]
    for pattern in patterns:
        match = re.search(pattern, s, flags=re.IGNORECASE)
        if match:
            try:
                val = int(match.group(1))
                if 0 < val < 1000000:
                    return val
            except Exception:
                continue
    return None

def _extract_prereg(text: str) -> Optional[str]:
    """Look for NCT numbers or registration mentions."""
    match = re.search(r"\bNCT\d{8}\b", text, flags=re.IGNORECASE)
    if match:
        return match.group(0)
    if re.search(r"trial registration|prospectively registered|clinicaltrials\.gov", text, flags=re.IGNORECASE):
        return "YES"
    return None

def _extract_p_value_info(text: str) -> Tuple[bool, bool]:
    """Look for p-values and confidence intervals."""
    has_p = bool(re.search(r"\bp\s*[<=>]\s*0?\.\d+", text, flags=re.IGNORECASE))
    has_ci = bool(re.search(r"95%\s*ci|confidence interval", text, flags=re.IGNORECASE))
    return has_p, has_ci

def compute_rps(item: Dict[str, Any]) -> float:
    """Compatibility wrapper for ReproducibilityScorer.compute()."""
    verbose = compute_rps_verbose(item)
    return verbose["rps"]

def compute_rps_verbose(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Standardized 4-feature RPS computation with backward-compatible return dict.
    
    Extracts features from the input dict (which may contain 'title', 'abstract', 
    'study_type', etc.) and maps them to the canonical ReproducibilityScorer.
    """
    text = " ".join([
        str(item.get("title", "")),
        str(item.get("abstract", "")),
        str(item.get("publication_types", "")),
    ])
    
    # Feature extraction
    design = _normalize_study_design(item.get("study_type", item.get("design", "")))
    sample_size = item.get("sample_size")
    if sample_size is None or (isinstance(sample_size, int) and sample_size <= 0):
        sample_size = _extract_sample_size(text)
    
    has_p, has_ci = _extract_p_value_info(text)
    if "p_value_present" in item:
        has_p = bool(item["p_value_present"])
    if "ci_present" in item:
        has_ci = bool(item["ci_present"])
        
    prereg = item.get("preregistration_id", _extract_prereg(text))
    
    # Construct canonical metadata
    metadata = StudyMetadata(
        sample_size=sample_size,
        study_design=design,
        has_p_value=has_p,
        has_CI=has_ci,
        preregistration_id=prereg,
        year=item.get("year") or item.get("publication_year")
    )
    
    # Standardised score
    rps = _SCORER.compute(metadata)
    
    # Map to compatibility dict (F14 fix: align 10-feature keys to 4-feature data)
    return {
        "rps": rps,
        "rps_raw": rps, # Legacy alias
        "rps_feature_coverage": 1.0, # Now always 1.0 because we have defaults for 4 features
        "available_features": {
            "study_design": design.value,
            "sample_size": sample_size,
            "p_value": has_p,
            "confidence_interval": has_ci,
            "preregistration": prereg
        },
        "missing_features": []
    }

def grade_from_rps(rps: float) -> str:
    """Standard grading: A >= 0.7, B >= 0.4, C < 0.4."""
    if rps >= 0.70: return "A"
    if rps >= 0.40: return "B"
    return "C"

# Deprecation warning
logger.warning("src.utils.rps_utils is deprecated. Use src.epistemic.reproducibility_scorer instead.")
