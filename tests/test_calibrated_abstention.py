from src.epistemic.calibrated_abstention import CalibratedAbstention
from src.models.enums import EpistemicState, ResponseTier


def test_falsified_state_overrides_low_belief_floor():
    abstainer = CalibratedAbstention(
        config={
            "ds": {"k_abstain": 0.8},
            "abstention": {
                "full_belief_min": 0.70,
                "min_belief_threshold": 0.10,
                "qualified_conflict_max": 0.50,
            },
        }
    )

    tier, rationale = abstainer.should_abstain(
        epistemic_state=EpistemicState.FALSIFIED,
        belief=0.0,
        uncertainty=0.0,
        conflict_K=0.0,
    )

    assert tier == ResponseTier.FULL
    assert "contradicts" in rationale
