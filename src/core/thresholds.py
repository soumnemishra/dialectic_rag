"""
Centralized threshold configuration for MA-RAG orchestration.

This module consolidates all hard-coded thresholds from across the system
(graph.py, controversy_classifier, etc.) into a single, validated, 
loadable configuration. Thresholds can be overridden via environment variables
or YAML configuration files.

Design Principles:
  - Single source of truth: all thresholds defined here
  - Validated ranges: 0.0-1.0 for scores, reasonable bounds for counts
  - Environment override: MRAGE_THRESH_* env vars take precedence
  - YAML fallback: loads from config/thresholds.yaml if present
  - Semantic grouping: temporal, conflict, polarity, etc.
"""

import logging
import os
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator
import yaml

logger = logging.getLogger(__name__)


class TemporalConflictThresholds(BaseModel):
    """Thresholds for Temporal Conflict Scoring (TCS) and decision logic."""
    
    tcs_trigger: float = Field(
        default=0.10,
        ge=0.0,
        le=1.0,
        description="TCS score above which dialectical retrieval is triggered"
    )
    tcs_max_pairs: int = Field(
        default=25,
        description="Maximum temporal pair comparisons per question"
    )
    min_year_gap: int = Field(
        default=2,
        description="Minimum year gap to consider as temporal conflict"
    )
    max_papers: int = Field(
        default=15,
        description="Maximum papers to scan for temporal conflicts"
    )


class ConflictDetectionThresholds(BaseModel):
    """Thresholds for detecting conflicting beliefs (Dempster-Shafer, etc.)."""
    
    ds_conflict_trigger: float = Field(
        default=0.08,
        ge=0.0,
        le=1.0,
        description="DS conflict score above which dialectical retrieval is triggered"
    )
    ds_uncertain_trigger: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="DS uncertainty threshold for gating additional retrieval"
    )
    step_consensus_min: float = Field(
        default=0.50,
        ge=0.0,
        le=1.0,
        description="Minimum step consensus (majority agreement) - below triggers dialectical"
    )
    conflicting_answers_threshold: int = Field(
        default=1,
        description="Number of distinct answer options above which conflict is detected"
    )


class ApplicabilityThresholds(BaseModel):
    """Thresholds for cohort applicability and external validity scoring."""
    
    applicability_trigger: float = Field(
        default=0.50,
        ge=0.0,
        le=1.0,
        description="Applicability score below which dialectical retrieval is triggered"
    )
    default_fallback: float = Field(
        default=0.50,
        ge=0.0,
        le=1.0,
        description="Default applicability when metadata is missing"
    )
    high_match_threshold: float = Field(
        default=0.80,
        ge=0.0,
        le=1.0,
        description="Applicability score threshold for 'high match' classification"
    )


class ReproducibilityThresholds(BaseModel):
    """Thresholds for Reproducibility Potential Score (RPS)."""
    
    rps_trigger: float = Field(
        default=0.20,
        ge=0.0,
        le=1.0,
        description="RPS score below which dialectical retrieval is triggered (with CONTESTED)"
    )
    rps_trigger_contested: bool = Field(
        default=True,
        description="Whether RPS trigger applies only when controversy is CONTESTED"
    )
    sample_size_max: int = Field(
        default=100000,
        description="Maximum sample size cap to avoid extractor hallucination"
    )
    continuous_scoring: bool = Field(
        default=True,
        description="Whether to use continuous RPS scoring vs. categorical"
    )


class ControversyThresholds(BaseModel):
    """Thresholds for classifying evidence as SETTLED, CONTESTED, or EVOLVING."""
    
    settled_rps_min: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Minimum RPS for SETTLED classification"
    )
    settled_applicability_min: float = Field(
        default=0.60,
        ge=0.0,
        le=1.0,
        description="Minimum applicability for SETTLED classification"
    )
    settled_confidence_min: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
        description="Minimum confidence for SETTLED classification"
    )
    settled_tcs_max: float = Field(
        default=0.20,
        ge=0.0,
        le=1.0,
        description="Maximum TCS for SETTLED classification"
    )
    contested_confidence_max: float = Field(
        default=0.60,
        ge=0.0,
        le=1.0,
        description="Maximum confidence for CONTESTED classification"
    )
    evolving_rps_max: float = Field(
        default=0.10,
        ge=0.0,
        le=1.0,
        description="Maximum RPS for EVOLVING classification"
    )
    evolving_tcs_min: float = Field(
        default=0.40,
        ge=0.0,
        le=1.0,
        description="Minimum TCS for EVOLVING classification"
    )


class DempsterShaferThresholds(BaseModel):
    """Thresholds for Dempster-Shafer belief fusion."""
    
    temporal_support_when_no_conflict: float = Field(
        default=0.50,
        ge=0.0,
        le=1.0,
        description="Support mass assigned when no temporal conflict"
    )
    rps_support_multiplier: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Multiplier applied to RPS when computing belief support"
    )
    max_support: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Cap on support to ensure uncertainty margin"
    )
    confidence_discount: float = Field(
        default=0.10,
        ge=0.0,
        le=1.0,
        description="Discount applied to confidence under uncertainty"
    )
    min_uncertainty: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Minimum uncertainty mass in belief interval"
    )


class ExtractionThresholds(BaseModel):
    """Thresholds for document chunk extraction and deduplication."""
    
    max_chunks_complex: int = Field(
        default=20,
        description="Maximum chunks to extract for complex questions"
    )
    max_chunks_moderate: int = Field(
        default=16,
        description="Maximum chunks to extract for moderate questions"
    )
    max_chunks_simple: int = Field(
        default=10,
        description="Maximum chunks to extract for simple questions"
    )
    dedup_threshold: float = Field(
        default=0.90,
        ge=0.0,
        le=1.0,
        description="Semantic similarity threshold for deduplication"
    )
    relevance_threshold: int = Field(
        default=2,
        description="Minimum lexical overlap score for a chunk to be considered relevant"
    )


class EvidencePolarityThresholds(BaseModel):
    """Thresholds for evidence polarity classification."""
    
    strong_support_min: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Minimum confidence for 'strong_support' classification"
    )
    weak_support_max: float = Field(
        default=0.55,
        ge=0.0,
        le=1.0,
        description="Maximum confidence for 'weak_support' classification"
    )
    refute_min: float = Field(
        default=0.60,
        ge=0.0,
        le=1.0,
        description="Minimum confidence for 'refute' classification"
    )


class RedundancyThresholds(BaseModel):
    """Thresholds for detecting redundant evidence and PMIDs."""
    
    pmid_overlap_threshold: float = Field(
        default=0.90,
        ge=0.0,
        le=1.0,
        description="Threshold for PMID overlap between retrieval steps above which diversity is forced"
    )


class RetryThresholds(BaseModel):
    """Thresholds for retry logic and supplemental retrieval."""
    
    retry_count_max: int = Field(
        default=1,
        description="Maximum retry attempts for supplemental retrieval"
    )
    force_accept_on_retry: int = Field(
        default=1,
        description="Force 'accept' decision when retry_count >= this value"
    )


class ThresholdConfig(BaseModel):
    """
    Master configuration container for all MA-RAG thresholds.
    
    This model is the single source of truth for decision thresholds across
    the entire orchestration pipeline. Organized into semantic groups for
    clarity and maintainability.
    """
    
    temporal_conflict: TemporalConflictThresholds = Field(
        default_factory=TemporalConflictThresholds,
        description="Temporal conflict scoring thresholds"
    )
    
    conflict_detection: ConflictDetectionThresholds = Field(
        default_factory=ConflictDetectionThresholds,
        description="Belief conflict and consensus thresholds"
    )
    
    applicability: ApplicabilityThresholds = Field(
        default_factory=ApplicabilityThresholds,
        description="Cohort applicability thresholds"
    )
    
    reproducibility: ReproducibilityThresholds = Field(
        default_factory=ReproducibilityThresholds,
        description="Reproducibility potential score thresholds"
    )
    
    controversy: ControversyThresholds = Field(
        default_factory=ControversyThresholds,
        description="Evidence controversy classification thresholds"
    )
    
    dempster_shafer: DempsterShaferThresholds = Field(
        default_factory=DempsterShaferThresholds,
        description="Dempster-Shafer belief fusion thresholds"
    )
    
    extraction: ExtractionThresholds = Field(
        default_factory=ExtractionThresholds,
        description="Document extraction thresholds"
    )
    
    evidence_polarity: EvidencePolarityThresholds = Field(
        default_factory=EvidencePolarityThresholds,
        description="Evidence polarity classification thresholds"
    )
    
    retry: RetryThresholds = Field(
        default_factory=RetryThresholds,
        description="Retry and supplemental retrieval thresholds"
    )
    
    redundancy: RedundancyThresholds = Field(
        default_factory=RedundancyThresholds,
        description="Evidence and PMID redundancy thresholds"
    )
    
    @field_validator("*", mode="before")
    @classmethod
    def validate_nested_models(cls, v):
        """Ensure nested models are properly instantiated."""
        return v
    
    @classmethod
    def load_from_yaml(cls, yaml_path: str) -> "ThresholdConfig":
        """Load thresholds from a YAML file.
        
        Args:
            yaml_path: Path to YAML config file
            
        Returns:
            ThresholdConfig instance
            
        Raises:
            FileNotFoundError: If yaml_path does not exist
            ValueError: If YAML is invalid
        """
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Threshold config file not found: {yaml_path}")
        
        try:
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f) or {}
            logger.info(f"Loaded threshold config from {yaml_path}")
            return cls(**data)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {yaml_path}: {e}")
        except Exception as e:
            raise ValueError(f"Error loading threshold config: {e}")
    
    @classmethod
    def from_env_overrides(cls, base_config: Optional["ThresholdConfig"] = None) -> "ThresholdConfig":
        """
        Create ThresholdConfig with environment variable overrides.
        
        Looks for MRAGE_THRESH_* environment variables and applies them
        on top of base_config or defaults.
        
        Supported env vars (examples):
          - MRAGE_THRESH_TCS_TRIGGER=0.15
          - MRAGE_THRESH_DS_CONFLICT_TRIGGER=0.10
          - MRAGE_THRESH_APPLICABILITY_TRIGGER=0.45
          - MRAGE_THRESH_STEP_CONSENSUS_MIN=0.60
          
        Args:
            base_config: Optional base config to override. Defaults to new instance.
            
        Returns:
            ThresholdConfig with environment overrides applied
        """
        config = base_config or cls()
        
        env_prefix = "MRAGE_THRESH_"
        overrides = {}
        
        for key, value in os.environ.items():
            if key.startswith(env_prefix):
                suffix = key[len(env_prefix):].lower()
                try:
                    # Try parsing as float
                    if '.' in value:
                        overrides[suffix] = float(value)
                    else:
                        # Try parsing as int
                        try:
                            overrides[suffix] = int(value)
                        except ValueError:
                            # Keep as string
                            overrides[suffix] = value
                except Exception as e:
                    logger.warning(f"Failed to parse {key}={value}: {e}")
        
        if overrides:
            logger.info(f"Applying {len(overrides)} environment threshold overrides")
            # Flatten nested updates into proper path structure
            nested_updates = cls._flatten_to_nested(overrides)
            config_dict = config.model_dump()
            config_dict = cls._deep_merge(config_dict, nested_updates)
            config = cls(**config_dict)
        
        return config
    
    @staticmethod
    def _flatten_to_nested(flat_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Convert flat keys (e.g., 'tcs_trigger') to nested structure."""
        nested = {}
        
        # Map flat keys to nested paths
        flat_to_nested = {
            "tcs_trigger": ("temporal_conflict", "tcs_trigger"),
            "tcs_max_pairs": ("temporal_conflict", "tcs_max_pairs"),
            "min_year_gap": ("temporal_conflict", "min_year_gap"),
            "max_papers": ("temporal_conflict", "max_papers"),
            "ds_conflict_trigger": ("conflict_detection", "ds_conflict_trigger"),
            "ds_uncertain_trigger": ("conflict_detection", "ds_uncertain_trigger"),
            "step_consensus_min": ("conflict_detection", "step_consensus_min"),
            "conflicting_answers_threshold": ("conflict_detection", "conflicting_answers_threshold"),
            "applicability_trigger": ("applicability", "applicability_trigger"),
            "default_fallback": ("applicability", "default_fallback"),
            "high_match_threshold": ("applicability", "high_match_threshold"),
            "rps_trigger": ("reproducibility", "rps_trigger"),
            "rps_trigger_contested": ("reproducibility", "rps_trigger_contested"),
            "sample_size_max": ("reproducibility", "sample_size_max"),
            "settled_rps_min": ("controversy", "settled_rps_min"),
            "settled_applicability_min": ("controversy", "settled_applicability_min"),
            "settled_confidence_min": ("controversy", "settled_confidence_min"),
            "settled_tcs_max": ("controversy", "settled_tcs_max"),
            "contested_confidence_max": ("controversy", "contested_confidence_max"),
            "evolving_rps_max": ("controversy", "evolving_rps_max"),
            "evolving_tcs_min": ("controversy", "evolving_tcs_min"),
            "pmid_overlap_threshold": ("redundancy", "pmid_overlap_threshold"),
            "relevance_threshold": ("extraction", "relevance_threshold"),
            "dedup_threshold": ("extraction", "dedup_threshold"),
        }
        
        for flat_key, value in flat_dict.items():
            if flat_key in flat_to_nested:
                parent, child = flat_to_nested[flat_key]
                if parent not in nested:
                    nested[parent] = {}
                nested[parent][child] = value
        
        return nested
    
    @staticmethod
    def _deep_merge(base_dict: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge updates into base_dict."""
        result = base_dict.copy()
        for key, value in updates.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = ThresholdConfig._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as nested dictionary."""
        return self.model_dump()


# Global singleton instance
_threshold_config: Optional[ThresholdConfig] = None


def get_threshold_config() -> ThresholdConfig:
    """
    Get or create the global threshold configuration singleton.
    
    On first call:
      1. Creates default ThresholdConfig
      2. Loads YAML overrides from config/thresholds.yaml (if exists)
      3. Applies environment variable overrides (MRAGE_THRESH_*)
    
    Returns:
        ThresholdConfig singleton
    """
    global _threshold_config
    
    if _threshold_config is not None:
        return _threshold_config
    
    # Start with defaults
    config = ThresholdConfig()
    
    # Load YAML if available
    yaml_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "config",
        "thresholds.yaml"
    )
    if os.path.exists(yaml_path):
        try:
            config = ThresholdConfig.load_from_yaml(yaml_path)
            logger.info(f"Loaded thresholds from {yaml_path}")
        except Exception as e:
            logger.warning(f"Failed to load threshold YAML: {e}, using defaults")
    
    # Apply environment overrides
    config = ThresholdConfig.from_env_overrides(config)
    
    _threshold_config = config
    return _threshold_config


def reset_threshold_config() -> None:
    """Reset the global singleton (useful for testing)."""
    global _threshold_config
    _threshold_config = None
