"""Temporal Belief Revision — Config-Driven Consensus Tracking.

Performs O(n) sequential belief updating over chronologically-ordered
evidence.  Enforces the *Two-Source Confirmation Rule*: a contradiction
only escalates the epistemic state if confirmed by ≥ ``min_confirming``
independent sources.

All NLI thresholds and reproducibility gates are loaded from
``config/default.yaml → nli`` and ``config/default.yaml → temporal``.
"""

import logging
from typing import List, Optional, Dict, Any

from src.models.schemas import EvidenceItem, EpistemicResult
from src.models.enums import EpistemicState, ResponseTier
from .nli_engine import NLIEngine
from src.config import epistemic_settings

logger = logging.getLogger(__name__)


class TemporalBeliefRevision:
    """Formal temporal epistemic reasoning component.

    Tracks consensus shifts with scientific conservatism.  Every tunable
    threshold is loaded from the YAML configuration to support ablation
    studies.

    Args:
        nli_engine: Optional pre-configured NLI engine.
        config: Optional override dict; defaults to the centralised
            ``epistemic_settings`` singleton.
    """

    def __init__(
        self,
        nli_engine: Optional[NLIEngine] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.nli_engine = nli_engine or NLIEngine()
        self.config = config if config is not None else epistemic_settings

        # --- Config-driven thresholds (F2 fix) ---
        nli_cfg = self.config.get("nli", {})
        temporal_cfg = self.config.get("temporal", {})

        self.contradiction_threshold: float = float(
            nli_cfg.get("contradiction_threshold", 0.80)
        )
        self.entailment_threshold: float = float(
            nli_cfg.get("entailment_threshold", 0.85)
        )
        self.min_reproducibility: float = float(
            nli_cfg.get("temporal_min_reproducibility", 0.50)
        )
        self.min_confirming: int = int(
            temporal_cfg.get("min_confirming_sources", 2)
        )

        logger.info(
            "TemporalBeliefRevision initialised: "
            "contradiction_threshold=%.2f, entailment_threshold=%.2f, "
            "min_reproducibility=%.2f, min_confirming=%d",
            self.contradiction_threshold,
            self.entailment_threshold,
            self.min_reproducibility,
            self.min_confirming,
        )

    async def detect_consensus_shift(
        self, evidences: List[EvidenceItem]
    ) -> EpistemicResult:
        """O(n) sequential belief updating.

        Enforces the *Two-Source Confirmation Rule* for belief revision:
        a contradiction only triggers state escalation if confirmed by
        ``self.min_confirming`` independent sources, each passing the
        ``self.min_reproducibility`` gate.

        Args:
            evidences: List of scored evidence items.

        Returns:
            EpistemicResult capturing the temporal analysis.
        """
        # 1. Sort evidence chronologically
        sorted_evidences = sorted(
            evidences,
            key=lambda x: (x.metadata.year or 0, -x.reproducibility_score),
        )

        if len(sorted_evidences) < 2:
            return EpistemicResult(
                state=EpistemicState.INSUFFICIENT,
                belief=0.0,
                uncertainty=1.0,
                conflict=0.0,
                temporal_shift_detected=False,
                response_tier=ResponseTier.ABSTAIN,
                evidence_items=evidences,
            )

        # 2. Select baseline premise (highest RPS among earliest evidence)
        baseline = sorted_evidences[0]
        current_belief = baseline.claim
        state = EpistemicState.SETTLED
        contradiction_events: List[Dict[str, Any]] = []

        # 3. Sequential Belief Updating with Confirmation Rule
        for i in range(1, len(sorted_evidences)):
            new_item = sorted_evidences[i]

            if not new_item.claim:
                continue

            nli_res = await self.nli_engine.classify(
                current_belief, new_item.claim
            )
            label = nli_res.get("label", "NEUTRAL")
            conf = nli_res.get("confidence", 0.0)

            # Contradiction gate: both NLI confidence AND study quality
            # must exceed config-driven thresholds.
            if (
                label == "CONTRADICTION"
                and conf > self.contradiction_threshold
            ):
                if new_item.reproducibility_score > self.min_reproducibility:
                    contradiction_events.append(
                        {
                            "pmid": new_item.pmid,
                            "year": new_item.metadata.year,
                            "reproducibility": new_item.reproducibility_score,
                            "claim": new_item.claim,
                        }
                    )

                    # Two-Source Confirmation check
                    if len(contradiction_events) >= self.min_confirming:
                        if state == EpistemicState.SETTLED:
                            state = EpistemicState.EVOLVING
                        elif state == EpistemicState.EVOLVING:
                            state = EpistemicState.CONTESTED

                    # Revise belief if new study is higher quality
                    if (
                        new_item.reproducibility_score
                        > baseline.reproducibility_score
                    ):
                        current_belief = new_item.claim

            elif (
                label == "ENTAILMENT"
                and conf > self.entailment_threshold
            ):
                if state == EpistemicState.INSUFFICIENT:
                    state = EpistemicState.SETTLED

        return EpistemicResult(
            state=state,
            belief=-1.0,  # Explicit out-of-bounds flag for temporal override
            uncertainty=-1.0,  # Explicit out-of-bounds flag for temporal override
            conflict=len(contradiction_events) / max(len(evidences), 1),
            temporal_shift_detected=(
                state in [EpistemicState.EVOLVING, EpistemicState.CONTESTED]
            ),
            response_tier=(
                ResponseTier.FULL
                if state == EpistemicState.SETTLED
                else ResponseTier.QUALIFIED
            ),
            evidence_items=evidences,
            baseline_claim=baseline.claim,
            current_belief=current_belief,
            contradiction_events=contradiction_events,
        )
