"""Reproducibility Potential Score (RPS) — Config-Driven 4-Feature Scorer.

Two-pass architecture:
    Pass 1: Extract grounded features (handled by MetadataExtractor).
    Pass 2: Deterministic scoring with YAML-configurable weights.

Mathematical formulation (aligned with thesis §Reproducibility):
    RPS = w_design · D + w_sample · S + w_pvalue · P + w_prereg · R

All four weights are loaded from ``config/default.yaml → reproducibility``
and validated to sum to 1.0 at init time.

Feature scoring:
    D (design_score):
        1.0 — meta-analysis, systematic review
        0.5 — RCT, cohort, case-control
        0.2 — case series, other
    S (sample_size_score):
        1.0 — N ≥ 1000
        0.5 — N ≥ 100
        0.2 — N < 100 or unknown
    P (pvalue_score):
        1.0 — reports both p-value and CI
        0.5 — reports p-value only
        0.2 — neither
    R (prereg_score):
        1.0 — has preregistration (NCT ID present)
        0.0 — no preregistration
        0.2 — unknown (conservative default)
"""

import logging
from typing import Optional, Dict, Any

from src.models.schemas import StudyMetadata, StudyDesign
from src.config import epistemic_settings

logger = logging.getLogger(__name__)


class ReproducibilityScorer:
    """Deterministic reproducibility scorer with config-driven weights.

    Loads all four weights from the ``reproducibility`` section of
    ``config/default.yaml`` and validates they sum to 1.0 (±tolerance).

    Args:
        config: Optional override dict; defaults to the centralised
            ``epistemic_settings`` singleton.
    """

    # Design score mapping — deterministic, no config needed
    DESIGN_SCORES: Dict[StudyDesign, float] = {
        StudyDesign.META_ANALYSIS: 1.0,
        StudyDesign.SYSTEMATIC_REVIEW: 1.0,
        StudyDesign.RCT: 0.5,
        StudyDesign.COHORT: 0.5,
        StudyDesign.CASE_CONTROL: 0.5,
        StudyDesign.CASE_SERIES: 0.2,
        StudyDesign.OTHER: 0.2,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config if config is not None else epistemic_settings
        weights_cfg = self.config.get("reproducibility", {})

        # --- Config-driven weights (F3 fix: added w_prereg) ---
        self.w_design: float = float(weights_cfg.get("w_design", 0.40))
        self.w_sample_size: float = float(weights_cfg.get("w_sample_size", 0.25))
        self.w_pvalue: float = float(weights_cfg.get("w_pvalue", 0.15))
        self.w_prereg: float = float(weights_cfg.get("w_prereg", 0.20))

        # --- Weight validation ---
        weight_sum = self.w_design + self.w_sample_size + self.w_pvalue + self.w_prereg
        if abs(weight_sum - 1.0) > 0.01:
            logger.warning(
                "Reproducibility weights do not sum to 1.0: "
                "w_design=%.2f + w_sample_size=%.2f + w_pvalue=%.2f + w_prereg=%.2f = %.4f",
                self.w_design, self.w_sample_size, self.w_pvalue, self.w_prereg,
                weight_sum,
            )

        logger.info(
            "ReproducibilityScorer initialised: w_design=%.2f, w_sample=%.2f, "
            "w_pvalue=%.2f, w_prereg=%.2f (sum=%.2f)",
            self.w_design, self.w_sample_size, self.w_pvalue, self.w_prereg,
            weight_sum,
        )

    def compute(self, metadata: StudyMetadata, return_components: bool = False) -> Any:
        """Compute the weighted Reproducibility Potential Score (RPS).

        Formula:
            RPS = w_design · D + w_sample · S + w_pvalue · P + w_prereg · R

        Args:
            metadata: Extracted study metadata.
            return_components: If True, returns a dict with RPS and component scores.

        Returns:
            RPS in [0, 1] (as float) or a dict.
        """
        # --- D: Study Design ---
        design_score = self.DESIGN_SCORES.get(metadata.study_design, 0.2)

        # --- S: Sample Size ---
        sample_size = metadata.sample_size or 0
        if sample_size >= 1000:
            sample_size_score = 1.0
        elif sample_size >= 100:
            sample_size_score = 0.5
        else:
            sample_size_score = 0.2

        # --- P: Statistical Reporting ---
        if metadata.has_p_value and getattr(metadata, "has_CI", False):
            pvalue_score = 1.0
        elif metadata.has_p_value:
            pvalue_score = 0.5
        else:
            pvalue_score = 0.2

        # --- R: Preregistration ---
        prereg_id = getattr(metadata, "preregistration_id", None)
        if prereg_id and str(prereg_id).strip().lower() not in ("", "none", "null"):
            prereg_score = 1.0
        elif prereg_id is None:
            prereg_score = 0.2  # Unknown — conservative default
        else:
            prereg_score = 0.0  # Explicitly no preregistration

        # --- Weighted sum ---
        rps = (
            self.w_design * design_score
            + self.w_sample_size * sample_size_score
            + self.w_pvalue * pvalue_score
            + self.w_prereg * prereg_score
        )

        # Clip to [0, 1]
        rps = max(0.0, min(1.0, rps))

        if return_components:
            return {
                "final_rps": rps,
                "components": {
                    "design_score": design_score,
                    "sample_size_score": sample_size_score,
                    "pvalue_score": pvalue_score,
                    "prereg_score": prereg_score
                },
                "weights": {
                    "w_design": self.w_design,
                    "w_sample_size": self.w_sample_size,
                    "w_pvalue": self.w_pvalue,
                    "w_prereg": self.w_prereg
                }
            }
        return rps
