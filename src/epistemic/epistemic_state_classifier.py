"""Epistemic State Classifier — Config-Driven Fuzzy Logic.

Maps final belief masses to epistemic states using fuzzy sigmoid
membership functions.  All midpoints and thresholds are loaded from
``config/default.yaml → states``.

States:
    SETTLED     — High belief, low uncertainty, low conflict
    CONTESTED   — High conflict
    INSUFFICIENT — High uncertainty
    EVOLVING    — Temporal shift detected (override)
    FALSIFIED   — m(false) dominates m(true) (config-driven thresholds)
"""

import logging
import math
from typing import Optional, Dict, Any

from src.models.enums import EpistemicState
from src.config import epistemic_settings

logger = logging.getLogger(__name__)


def sigmoid(x: float, mid: float, slope: float) -> float:
    """Sigmoid ramp function for fuzzy membership.

    Args:
        x: Input value.
        mid: Midpoint of the sigmoid.
        slope: Steepness of the transition.

    Returns:
        Membership value in (0, 1).
    """
    return 1.0 / (1.0 + math.exp(-slope * (x - mid)))


class EpistemicStateClassifier:
    """Fuzzy-logic classifier for epistemic state assignment.

    All parameters are loaded from the ``states`` section of
    ``config/default.yaml``.

    Args:
        config: Optional override dict; defaults to the centralised
            ``epistemic_settings`` singleton.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config if config is not None else epistemic_settings
        self.state_config = self.config.get("states", {})

        # F9 fix: load FALSIFIED thresholds from config instead of hardcoding
        falsified_cfg = self.state_config.get("falsified", {})
        self.falsified_belief_false_min: float = float(
            falsified_cfg.get("belief_false_min", 0.60)
        )
        self.falsified_belief_true_max: float = float(
            falsified_cfg.get("belief_true_max", 0.10)
        )

        logger.info(
            "EpistemicStateClassifier initialised: "
            "falsified_bf_min=%.2f, falsified_bt_max=%.2f",
            self.falsified_belief_false_min,
            self.falsified_belief_true_max,
        )

    def classify(
        self,
        belief: float,
        uncertainty: float,
        conflict: float,
        temporal_shift_detected: bool,
        belief_true: float = 0.0,
        belief_false: float = 0.0,
    ) -> EpistemicState:
        """Classify the epistemic state using fuzzy membership logic.

        Decision order:
            1. EVOLVING — if temporal shift detected (hard override)
            2. FALSIFIED — if m(false) > threshold AND m(true) < threshold
            3. max(μ_SETTLED, μ_CONTESTED, μ_INSUFFICIENT)

        Args:
            belief: Pignistic probability P(true).
            uncertainty: m(Θ) — mass on the full frame.
            conflict: Global conflict K.
            temporal_shift_detected: Whether temporal analysis found a shift.
            belief_true: m(true) from DS fusion.
            belief_false: m(false) from DS fusion.

        Returns:
            The classified EpistemicState.
        """
        if temporal_shift_detected:
            return EpistemicState.EVOLVING

        # Strong Opposition / Falsification Detection (F9: config-driven)
        if (
            belief_false > self.falsified_belief_false_min
            and belief_true < self.falsified_belief_true_max
        ):
            return EpistemicState.FALSIFIED

        cfg = self.state_config

        # Membership in SETTLED
        # μ_SETTLED = σ(belief) · σ(-uncertainty) · σ(-conflict)
        s_cfg = cfg.get("settled", {})
        u_settled = (
            sigmoid(belief, mid=s_cfg.get("belief_mid", 0.70), slope=10)
            * sigmoid(-uncertainty, mid=-s_cfg.get("uncertainty_max", 0.20), slope=15)
            * sigmoid(-conflict, mid=-s_cfg.get("conflict_max", 0.20), slope=15)
        )

        # Membership in CONTESTED
        c_cfg = cfg.get("contested", {})
        u_contested = sigmoid(
            conflict, mid=c_cfg.get("conflict_mid", 0.40), slope=10
        )

        # Membership in INSUFFICIENT
        i_cfg = cfg.get("insufficient", {})
        u_insufficient = sigmoid(
            uncertainty, mid=i_cfg.get("uncertainty_mid", 0.50), slope=10
        )

        # Select state with highest membership
        memberships = {
            EpistemicState.SETTLED: u_settled,
            EpistemicState.CONTESTED: u_contested,
            EpistemicState.INSUFFICIENT: u_insufficient,
        }

        # Hardening v2: Prioritise CONTESTED over INSUFFICIENT if conflict is high
        # This ensures that "fundamentally split" evidence is not reported as "ignorance"
        if conflict > cfg.get("contested", {}).get("conflict_mid", 0.40) and u_contested >= u_insufficient:
             return EpistemicState.CONTESTED

        return max(memberships, key=memberships.get)
