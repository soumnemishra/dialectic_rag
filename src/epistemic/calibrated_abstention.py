import logging
import yaml
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from src.models.enums import EpistemicState, ResponseTier

logger = logging.getLogger(__name__)

class CalibratedAbstention:
    """
    Three-tier response model: FULL, QUALIFIED, ABSTAIN.
    """

    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.ds_config = self.config.get("ds", {})
        self.abs_config = self.config.get("abstention", {})

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        path = config_path or Path(__file__).resolve().parents[1] / "config" / "default.yaml"
        try:
            with open(path, "r") as f:
                return yaml.safe_load(f)
        except Exception:
            return {
                "ds": {"k_abstain": 0.8},
                "abstention": {"full_belief_min": 0.70}
            }

    def should_abstain(
        self,
        epistemic_state: EpistemicState,
        belief: float,
        uncertainty: float,
        conflict_K: float
    ) -> Tuple[ResponseTier, str]:
        """
        Decision logic for calibrated abstention.
        """
        k_abstain = self.ds_config.get("k_abstain", 0.8)
        belief_min = self.abs_config.get("full_belief_min", 0.70)

        # 0. Falsified state (Evidence strongly contradicts the hypothesis)
        if epistemic_state == EpistemicState.FALSIFIED:
            return ResponseTier.FULL, "Strong evidence consistently contradicts the original clinical hypothesis."

        # 1. Irreconcilable conflict
        if conflict_K >= k_abstain:
            return ResponseTier.ABSTAIN, "Irreconcilable evidence conflict detected."

        # 2. Insufficient evidence
        if epistemic_state == EpistemicState.INSUFFICIENT:
            return ResponseTier.ABSTAIN, "Insufficient high-quality evidence available to form a conclusion."

        # 3. Contested evidence
        if epistemic_state == EpistemicState.CONTESTED:
            return ResponseTier.QUALIFIED, "Evidence is significantly contested; presenting a balanced view of conflicting findings."

        # 4. Evolving evidence
        if epistemic_state == EpistemicState.EVOLVING:
            return ResponseTier.QUALIFIED, "Scientific consensus is evolving; recent high-quality findings suggest a shift in understanding."

        # 5. Settled but low belief
        if epistemic_state == EpistemicState.SETTLED and belief < belief_min:
            return ResponseTier.QUALIFIED, "Evidence appears settled but lacks the high-confidence threshold for a definitive answer."

        # 5. Low belief floor
        min_belief_threshold = self.abs_config.get("min_belief_threshold", 0.1)
        if belief < min_belief_threshold:
            return ResponseTier.ABSTAIN, f"Aggregated belief mass ({belief:.3f}) is below the minimum threshold ({min_belief_threshold}) for clinical assertion."

        # 6. Settled and high belief
        if epistemic_state == EpistemicState.SETTLED and belief >= belief_min:
            return ResponseTier.FULL, "Sufficient high-quality evidence supports a settled clinical conclusion."

        return ResponseTier.QUALIFIED, "Evidence status requires a qualified clinical assessment."
