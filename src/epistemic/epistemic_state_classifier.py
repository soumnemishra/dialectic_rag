import logging
import yaml
import math
from pathlib import Path
from typing import Optional, Dict, Any
from src.models.enums import EpistemicState

logger = logging.getLogger(__name__)

def sigmoid(x, mid, slope):
    """Sigmoid ramp function for fuzzy membership."""
    return 1 / (1 + math.exp(-slope * (x - mid)))

class EpistemicStateClassifier:
    """
    Maps final masses to epistemic states using fuzzy membership.
    """

    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.state_config = self.config.get("states", {})

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        path = config_path or Path(__file__).resolve().parents[1] / "config" / "default.yaml"
        try:
            with open(path, "r") as f:
                return yaml.safe_load(f)
        except Exception:
            return {
                "states": {
                    "settled": {"belief_mid": 0.70, "uncertainty_max": 0.20, "conflict_max": 0.20},
                    "contested": {"conflict_mid": 0.40},
                    "insufficient": {"uncertainty_mid": 0.50}
                }
            }

    def classify(
        self,
        belief: float,
        uncertainty: float,
        conflict: float,
        temporal_shift_detected: bool,
        belief_true: float = 0.0,
        belief_false: float = 0.0
    ) -> EpistemicState:
        """
        Fuzzy membership logic for epistemic state classification.
        """
        if temporal_shift_detected:
            return EpistemicState.EVOLVING

        # Strong Opposition / Falsification Detection
        # If belief_false is high and belief_true is low, it's FALSIFIED
        if belief_false > 0.6 and belief_true < 0.1:
            return EpistemicState.FALSIFIED

        cfg = self.state_config
        
        # Membership in SETTLED
        # μ_SETTLED = sigmoid(belief) * sigmoid(-uncertainty) * sigmoid(-conflict)
        s_cfg = cfg.get("settled", {})
        u_settled = (
            sigmoid(belief, mid=s_cfg.get("belief_mid", 0.70), slope=10) *
            sigmoid(-uncertainty, mid=-s_cfg.get("uncertainty_max", 0.20), slope=15) *
            sigmoid(-conflict, mid=-s_cfg.get("conflict_max", 0.20), slope=15)
        )
        
        # Membership in CONTESTED
        c_cfg = cfg.get("contested", {})
        u_contested = sigmoid(conflict, mid=c_cfg.get("conflict_mid", 0.40), slope=10)
        
        # Membership in INSUFFICIENT
        i_cfg = cfg.get("insufficient", {})
        u_insufficient = sigmoid(uncertainty, mid=i_cfg.get("uncertainty_mid", 0.50), slope=10)
        
        # Select state with highest membership
        memberships = {
            EpistemicState.SETTLED: u_settled,
            EpistemicState.CONTESTED: u_contested,
            EpistemicState.INSUFFICIENT: u_insufficient
        }
        
        return max(memberships, key=memberships.get)
