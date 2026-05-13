"""Dempster-Shafer helper utilities for epistemic fusion.
Strictly aligned with the DIALECTIC-RAG blueprint.
"""
import logging
import yaml
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from src.models.enums import EvidenceStance, EpistemicState
from src.models.schemas import MassFunction, EvidenceItem

logger = logging.getLogger(__name__)

class DempsterShaferIntegrator:
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        ds_config = self.config.get("ds", {})
        self.gamma = ds_config.get("gamma", 0.9)
        self.k_threshold = ds_config.get("k_threshold", 0.5)
        self.k_abstain = ds_config.get("k_abstain", 0.8)

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        path = config_path or Path(__file__).resolve().parents[1] / "config" / "default.yaml"
        try:
            with open(path, "r") as f:
                return yaml.safe_load(f)
        except Exception:
            return {"ds": {"gamma": 0.9, "k_threshold": 0.5, "k_abstain": 0.8}}

    def assign_mass(self, evidence: EvidenceItem) -> MassFunction:
        """
        Assign belief masses based on evidence stance, quality, and applicability.
        w = reproducibility_score * applicability_score
        """
        # 1. Base weight calculation
        w_raw = evidence.reproducibility_score * evidence.applicability_score
        contr_prob = evidence.nli_contradiction_prob or 0.0
        stance = evidence.stance or EvidenceStance.NEUTRAL
        
        # Enforce minimum weight for stance-bearing evidence
        min_weight = self.config.get("evidence_gating", {}).get("min_weight", 0.05)
        w = w_raw
        if stance in [EvidenceStance.SUPPORT, EvidenceStance.OPPOSE, EvidenceStance.REFINE]:
            w = max(w_raw, min_weight)

        # 2. Assignment logic
        belief_true = 0.0
        belief_false = 0.0
        min_mass_floor = self.config.get("ds", {}).get("min_belief_mass", 0.05)
        
        if stance == EvidenceStance.SUPPORT:
            belief_true = self.gamma * w * (1 - contr_prob)
            if w > 0.1: belief_true = max(belief_true, min_mass_floor)
        elif stance == EvidenceStance.OPPOSE:
            belief_false = self.gamma * w * contr_prob
            if w > 0.1: belief_false = max(belief_false, min_mass_floor)
        elif stance == EvidenceStance.REFINE:
            belief_true = self.gamma * w * (1 - contr_prob) * 0.7
            if w > 0.1: belief_true = max(belief_true, min_mass_floor * 0.5)
        
        # Uncertainty is the remaining mass
        uncertainty = 1.0 - belief_true - belief_false
        
        return MassFunction(
            belief_true=max(0.0, float(belief_true)),
            belief_false=max(0.0, float(belief_false)),
            uncertainty=max(0.0, float(uncertainty))
        )

    def combine(self, m1: MassFunction, m2: MassFunction) -> MassFunction:
        """
        Combine two mass functions using Dempster's rule (with Yager fallback).
        """
        s1, r1, u1 = m1.belief_true, m1.belief_false, m1.uncertainty
        s2, r2, u2 = m2.belief_true, m2.belief_false, m2.uncertainty

        # Conflict coefficient K
        k = (s1 * r2) + (r1 * s2)

        # Pairwise combinations
        support = (s1 * s2) + (s1 * u2) + (u1 * s2)
        refute = (r1 * r2) + (r1 * u2) + (u1 * r2)
        uncertain = (u1 * u2)

        if k >= self.k_threshold:
            # Yager rule: assign conflict mass to universal set (uncertainty)
            uncertain += k
            total = support + refute + uncertain
            return MassFunction(
                belief_true=support / total,
                belief_false=refute / total,
                uncertainty=uncertain / total,
                conflict_K=float(k)
            )

        # Standard Dempster's rule
        denom = 1.0 - k
        if denom <= 0: # Pure conflict
            return MassFunction(belief_true=0.0, belief_false=0.0, uncertainty=1.0, conflict_K=1.0)
            
        return MassFunction(
            belief_true=support / denom,
            belief_false=refute / denom,
            uncertainty=uncertain / denom,
            conflict_K=float(k)
        )

    def pignistic_probability(self, mass: MassFunction) -> float:
        """Convert belief function to probability: P(true) = belief_true + 0.5 * uncertainty."""
        return round(mass.belief_true + 0.5 * mass.uncertainty, 3)

    def fuse_pool(self, pool: List[EvidenceItem]) -> Tuple[MassFunction, float]:
        """Combine all evidence in the pool."""
        combined = MassFunction(belief_true=0.0, belief_false=0.0, uncertainty=1.0)
        max_k = 0.0
        
        for item in pool:
            m = self.assign_mass(item)
            combined = self.combine(combined, m)
            if combined.conflict_K:
                max_k = max(max_k, combined.conflict_K)
                
        return combined, max_k
