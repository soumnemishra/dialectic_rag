"""Calibrated Abstention — Config-Driven Response Tier Assignment.

Three-tier response model:
    FULL      — Confident assertion with full evidence support.
    QUALIFIED — Hedged assertion with caveats about conflict or evolution.
    ABSTAIN   — Insufficient or irreconcilable evidence.

All decision thresholds are loaded from ``config/default.yaml``:
    - ``ds.k_abstain``           — irreconcilable conflict threshold
    - ``abstention.full_belief_min``       — belief floor for FULL tier
    - ``abstention.min_belief_threshold``  — absolute belief floor for any assertion
    - ``abstention.qualified_conflict_max`` — conflict ceiling for QUALIFIED→FULL
"""

import logging
from typing import Optional, Dict, Any, Tuple

from src.models.enums import EpistemicState, ResponseTier
from src.config import epistemic_settings

logger = logging.getLogger(__name__)


class CalibratedAbstention:
    """Three-tier response model with config-driven thresholds.

    Args:
        config: Optional override dict; defaults to the centralised
            ``epistemic_settings`` singleton.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config if config is not None else epistemic_settings
        self.ds_config = self.config.get("ds", {})
        self.abs_config = self.config.get("abstention", {})

        # All thresholds loaded from config (F4/F5 fixes)
        self.k_abstain: float = float(
            self.ds_config.get("k_abstain", 0.8)
        )
        self.full_belief_min: float = float(
            self.abs_config.get("full_belief_min", 0.70)
        )
        self.min_belief_threshold: float = float(
            self.abs_config.get("min_belief_threshold", 0.10)
        )
        # F5 fix: this was defined in YAML but never loaded
        self.qualified_conflict_max: float = float(
            self.abs_config.get("qualified_conflict_max", 0.50)
        )

        logger.info(
            "CalibratedAbstention initialised: k_abstain=%.2f, "
            "full_belief_min=%.2f, min_belief=%.2f, qualified_conflict_max=%.2f",
            self.k_abstain, self.full_belief_min,
            self.min_belief_threshold, self.qualified_conflict_max,
        )

    def should_abstain(
        self,
        epistemic_state: EpistemicState,
        belief: float,
        uncertainty: float,
        conflict_K: float,
    ) -> Tuple[ResponseTier, str]:
        """Decision logic for calibrated abstention.

        Decision cascade (order matters):
            0.  belief < min_belief_threshold   → ABSTAIN
            0.1 FALSIFIED                       → FULL (negative conclusion)
            1.  conflict ≥ k_abstain            → ABSTAIN
            2.  INSUFFICIENT                    → ABSTAIN
            3.  CONTESTED                       → QUALIFIED
            4.  EVOLVING                        → QUALIFIED
            5.  SETTLED + low belief            → QUALIFIED
            6.  SETTLED + high belief + low conflict → FULL
            7.  Fallback                        → QUALIFIED

        Args:
            epistemic_state: Classified epistemic state.
            belief: Pignistic probability P(true).
            uncertainty: m(Θ) mass.
            conflict_K: Global conflict K.

        Returns:
            Tuple of (ResponseTier, human-readable rationale string).
        """
        # 0. Low belief floor (Absolute threshold — F4: now from config)
        if belief < self.min_belief_threshold:
            return (
                ResponseTier.ABSTAIN,
                f"Aggregated belief mass ({belief:.3f}) is below the minimum "
                f"threshold ({self.min_belief_threshold}) for clinical assertion.",
            )

        # 0.1. Falsified state (Evidence strongly contradicts the hypothesis)
        if epistemic_state == EpistemicState.FALSIFIED:
            return (
                ResponseTier.FULL,
                "Strong evidence consistently contradicts the original "
                "clinical hypothesis.",
            )

        # 1. Irreconcilable conflict
        if conflict_K >= self.k_abstain:
            return (
                ResponseTier.ABSTAIN,
                "Irreconcilable evidence conflict detected.",
            )

        # 2. Insufficient evidence
        if epistemic_state == EpistemicState.INSUFFICIENT:
            return (
                ResponseTier.ABSTAIN,
                "Insufficient high-quality evidence available to form "
                "a conclusion.",
            )

        # 3. Contested evidence
        if epistemic_state == EpistemicState.CONTESTED:
            return (
                ResponseTier.QUALIFIED,
                "Evidence is significantly contested; presenting a "
                "balanced view of conflicting findings.",
            )

        # 4. Evolving evidence
        if epistemic_state == EpistemicState.EVOLVING:
            return (
                ResponseTier.QUALIFIED,
                "Scientific consensus is evolving; recent high-quality "
                "findings suggest a shift in understanding.",
            )

        # 5. Settled but low belief
        if (
            epistemic_state == EpistemicState.SETTLED
            and belief < self.full_belief_min
        ):
            return (
                ResponseTier.QUALIFIED,
                "Evidence appears settled but lacks the high-confidence "
                "threshold for a definitive answer.",
            )

        # 6. Settled, high belief, AND conflict below the QUALIFIED ceiling
        #    (F5 fix: qualified_conflict_max now actually used)
        if (
            epistemic_state == EpistemicState.SETTLED
            and belief >= self.full_belief_min
            and conflict_K <= self.qualified_conflict_max
        ):
            return (
                ResponseTier.FULL,
                "Sufficient high-quality evidence supports a settled "
                "clinical conclusion.",
            )

        return (
            ResponseTier.QUALIFIED,
            "Evidence status requires a qualified clinical assessment.",
        )
