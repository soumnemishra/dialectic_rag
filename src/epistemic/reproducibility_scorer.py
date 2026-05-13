import logging
import math
import yaml
from pathlib import Path
from typing import Optional, Dict, Any
from src.models.schemas import StudyMetadata, StudyDesign

logger = logging.getLogger(__name__)

class ReproducibilityScorer:
    """
    Two-pass architecture.
    Pass 1: Extract grounded features (handled by MetadataExtractor).
    Pass 2: Deterministic scoring with configurable weights.
    
    RPS = w1 * design_score + w2 * sample_size_score + w3 * pvalue_score
    w1 = 0.5, w2 = 0.3, w3 = 0.2
    
    design_score: 
      1.0 for meta-analysis, systematic review
      0.5 for RCT, cohort, case-control
      0.2 for other (case report, narrative)
      
    sample_size_score:
      1.0 for N >= 1000
      0.5 for N >= 100
      0.2 for N < 100 or unknown
      
    pvalue_score:
      1.0 for reporting both p-value and CI
      0.5 for reporting p-value only
      0.2 for no p-value or none
    """

    # Design score mapping
    DESIGN_SCORES = {
        StudyDesign.META_ANALYSIS: 1.0,
        StudyDesign.SYSTEMATIC_REVIEW: 1.0,
        StudyDesign.RCT: 0.5,
        StudyDesign.COHORT: 0.5,
        StudyDesign.CASE_CONTROL: 0.5,
        StudyDesign.CASE_SERIES: 0.2,
        StudyDesign.OTHER: 0.2,
    }

    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.weights = self.config.get("reproducibility", {})
        
        # Load weights, defaulting to the paper's specified values
        self.w1 = self.weights.get("w_design", 0.5)
        self.w2 = self.weights.get("w_sample_size", 0.3)
        self.w3 = self.weights.get("w_pvalue", 0.2)

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        path = config_path or Path(__file__).resolve().parents[1] / "config" / "default.yaml"
        try:
            with open(path, "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Failed to load config from {path}: {e}. Using defaults.")
            return {}

    def compute(self, metadata: StudyMetadata) -> float:
        """Compute the weighted Reproducibility Potential Score (RPS)."""
        design_score = self.DESIGN_SCORES.get(metadata.study_design, 0.2)
        
        sample_size = metadata.sample_size or 0
        if sample_size >= 1000:
            sample_size_score = 1.0
        elif sample_size >= 100:
            sample_size_score = 0.5
        else:
            sample_size_score = 0.2
            
        if metadata.has_p_value and getattr(metadata, "has_CI", False):
            pvalue_score = 1.0
        elif metadata.has_p_value:
            pvalue_score = 0.5
        else:
            pvalue_score = 0.2
            
        rps = (self.w1 * design_score) + (self.w2 * sample_size_score) + (self.w3 * pvalue_score)
        
        # Clip to [0, 1]
        return max(0.0, min(1.0, rps))
