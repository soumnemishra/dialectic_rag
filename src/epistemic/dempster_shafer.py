"""Dempster-Shafer Evidence Fusion — Config-Driven.

Implements belief-function combination for the DIALECTIC-RAG epistemic
engine.  Strictly aligned with the DIALECTIC-RAG blueprint.

Mathematical formulation (aligned with thesis §Evidence Fusion):

    Pairwise combination follows standard Dempster's Rule:
        K       = m1(true)·m2(false) + m1(false)·m2(true)
        m(true) = [m1(true)·m2(true) + m1(true)·m2(Θ) + m1(Θ)·m2(true)] / (1-K)
        m(false)= [m1(false)·m2(false)+ m1(false)·m2(Θ)+ m1(Θ)·m2(false)]/ (1-K)
        m(Θ)    = m1(Θ)·m2(Θ) / (1-K)

    Global conflict uses an empirical heuristic (see Thesis Note below):
        K_global = α · (Σm_s · Σm_r) / (Σm_s + Σm_r)²
    where α = ``conflict_scaling_factor`` from config.

    THESIS NOTE — Heuristic Global Conflict Approximation:
        Standard Dempster-Shafer theory computes conflict as the
        accumulated empty-set mass (∅) from sequential pairwise
        combination.  When combining 10+ medical studies with even
        slight contradictions, standard DS combination drives K → 1.0
        (Zadeh's paradox), collapsing the denominator.  The heuristic
        approximation above avoids this degenerate behaviour while
        preserving the conflict signal.  The scaling factor α is
        empirically tuned and configurable via YAML for ablation.

All parameters loaded from ``config/default.yaml → ds``.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from src.models.enums import EvidenceStance
from src.models.schemas import MassFunction, EvidenceItem
from src.config import epistemic_settings

logger = logging.getLogger(__name__)


class DempsterShaferIntegrator:
    """Dempster-Shafer belief fusion engine.

    All tunable parameters are loaded from the ``ds`` section of
    ``config/default.yaml``.

    Args:
        config: Optional override dict; defaults to the centralised
            ``epistemic_settings`` singleton.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config if config is not None else epistemic_settings
        ds_config = self.config.get("ds", {})

        self.gamma: float = float(ds_config.get("gamma", 0.9))
        self.k_threshold: float = float(ds_config.get("k_threshold", 0.5))
        self.k_abstain: float = float(ds_config.get("k_abstain", 0.8))
        self.conflict_scaling_factor: float = float(
            ds_config.get("conflict_scaling_factor", 4.0)
        )
        # F6 fix: was hardcoded as 0.7 — now loaded from YAML
        self.refine_discount: float = float(
            ds_config.get("refine_discount", 0.70)
        )

        logger.info(
            "DempsterShaferIntegrator initialised: gamma=%.2f, k_threshold=%.2f, "
            "k_abstain=%.2f, conflict_scaling=%.1f, refine_discount=%.2f",
            self.gamma, self.k_threshold, self.k_abstain,
            self.conflict_scaling_factor, self.refine_discount,
        )

    def assign_mass(self, evidence: EvidenceItem) -> MassFunction:
        """Assign belief masses based on evidence quality (RPS × Applicability).

        NLI contradiction is used to modulate uncertainty, not squash
        the primary stance.

        Args:
            evidence: A scored evidence item.

        Returns:
            MassFunction with belief_true, belief_false, uncertainty.
        """
        # Sanitise inputs to prevent NoneType crashes (Hardening v2)
        rps = getattr(evidence, 'reproducibility_score', 0.0) or 0.0
        app = getattr(evidence, 'applicability_score', 0.0) or 0.0
        w = max(rps * app, 0.0)
        
        stance = evidence.stance or EvidenceStance.NEUTRAL

        # Enforce minimum weight for stance-bearing evidence
        min_weight = self.config.get("evidence_gating", {}).get("min_weight", 0.05)
        if stance in [EvidenceStance.SUPPORT, EvidenceStance.OPPOSE]:
            w = max(w, min_weight)

        belief_true, belief_false = 0.0, 0.0

        # Direct assignment based on stance
        if stance == EvidenceStance.SUPPORT:
            belief_true = self.gamma * w
        elif stance == EvidenceStance.OPPOSE:
            belief_false = self.gamma * w
        elif stance == EvidenceStance.REFINE:
            # Partial support — discount loaded from config (F6 fix)
            belief_true = self.gamma * w * self.refine_discount

        # Uncertainty captures whatever mass is left
        uncertainty = max(1.0 - belief_true - belief_false, 0.0)

        return MassFunction(
            belief_true=float(belief_true),
            belief_false=float(belief_false),
            uncertainty=float(uncertainty),
        )

    def combine(self, m1: MassFunction, m2: MassFunction) -> MassFunction:
        """Standard Dempster's Rule — pairwise combination.

        Maintains mathematical associativity across the evidence pool.

        Args:
            m1: First mass function.
            m2: Second mass function.

        Returns:
            Combined MassFunction with pairwise conflict_K.
        """
        s1, r1, u1 = m1.belief_true, m1.belief_false, m1.uncertainty
        s2, r2, u2 = m2.belief_true, m2.belief_false, m2.uncertainty

        # Conflict coefficient K
        k = (s1 * r2) + (r1 * s2)

        # Pairwise combinations
        support = (s1 * s2) + (s1 * u2) + (u1 * s2)
        refute = (r1 * r2) + (r1 * u2) + (u1 * r2)
        uncertain = u1 * u2

        # Pure conflict edge case (e.g., 100% True vs 100% False)
        denom = 1.0 - k
        if denom <= 0:
            return MassFunction(
                belief_true=0.0, belief_false=0.0,
                uncertainty=1.0, conflict_K=1.0,
            )

        # Standard normalization
        return MassFunction(
            belief_true=support / denom,
            belief_false=refute / denom,
            uncertainty=uncertain / denom,
            conflict_K=float(k),  # Pairwise K, not global
        )

    def pignistic_probability(self, mass: MassFunction) -> float:
        """Convert belief function to probability.

        BetP(true) = Bel(true) + 0.5 · m(Θ)

        Args:
            mass: Combined mass function.

        Returns:
            Pignistic probability P(true).
        """
        return round(mass.belief_true + 0.5 * mass.uncertainty, 3)

    def fuse_pool(
        self, pool: List[EvidenceItem]
    ) -> Tuple[MassFunction, float]:
        """Combine all evidence and calculate global conflict.

        HEURISTIC NOTE (document in thesis):
            Global conflict is approximated as:
                K_global = α · (Σm_s · Σm_r) / (Σm_s + Σm_r)²
            where α = ``conflict_scaling_factor`` (default 4.0).
            This avoids the K→1 degeneration (Zadeh's paradox) that
            occurs with standard sequential DS combination over large
            evidence pools.

        Args:
            pool: List of scored evidence items.

        Returns:
            Tuple of (combined MassFunction, global conflict K).
        """
        if not pool:
            return (
                MassFunction(belief_true=0.0, belief_false=0.0, uncertainty=1.0),
                0.0,
            )

        # Start with a completely uncertain prior
        combined = MassFunction(
            belief_true=0.0, belief_false=0.0, uncertainty=1.0
        )

        # Un-normalized mass tracking for Global K heuristic
        sum_support = 0.0
        sum_refute = 0.0

        for item in pool:
            m = self.assign_mass(item)
            combined = self.combine(combined, m)

            # Accumulate for global conflict approximation
            if item.stance == EvidenceStance.SUPPORT:
                sum_support += m.belief_true
            elif item.stance == EvidenceStance.OPPOSE:
                sum_refute += m.belief_false

        # Global conflict heuristic (see docstring above)
        global_k = (sum_support * sum_refute) / max(
            (sum_support + sum_refute) ** 2, 1e-9
        )

        # Scale to [0,1] using config-driven scaling factor
        global_k_scaled = min(global_k * self.conflict_scaling_factor, 1.0)

        return combined, global_k_scaled
