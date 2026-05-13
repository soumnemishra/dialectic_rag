import logging
from typing import List, Optional, Dict, Any
from src.models.schemas import EvidenceItem, EpistemicResult
from src.models.enums import EpistemicState, ResponseTier
from .nli_engine import NLIEngine

logger = logging.getLogger(__name__)

class TemporalBeliefRevision:
    """
    Formal temporal epistemic reasoning component.
    Tracks consensus shifts with scientific conservatism.
    """
    
    def __init__(self, nli_engine: Optional[NLIEngine] = None, config_path: Optional[str] = None):
        self.nli_engine = nli_engine or NLIEngine()
        self.config = self._load_config(config_path)
        self.min_confirming = self.config.get("temporal", {}).get("min_confirming_sources", 2)

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        from pathlib import Path
        import yaml
        path = config_path or Path(__file__).resolve().parents[1] / "config" / "default.yaml"
        try:
            with open(path, "r") as f:
                return yaml.safe_load(f)
        except Exception:
            return {"temporal": {"min_confirming_sources": 2}}

    async def detect_consensus_shift(self, evidences: List[EvidenceItem]) -> EpistemicResult:
        """
        O(n) sequential belief updating.
        Enforces the 'Two-Source Confirmation' rule for belief revision.
        """
        # 1. Sort evidence chronologically
        sorted_evidences = sorted(evidences, key=lambda x: (x.metadata.year or 0, -x.reproducibility_score))
        
        if len(sorted_evidences) < 2:
            return EpistemicResult(
                state=EpistemicState.INSUFFICIENT,
                belief=0.0,
                uncertainty=1.0,
                conflict=0.0,
                temporal_shift_detected=False,
                response_tier=ResponseTier.ABSTAIN,
                evidence_items=evidences
            )

        # 2. Select baseline premise
        # Rule: Prefer highest reproducibility score among earliest evidence.
        baseline = sorted_evidences[0]
        current_belief = baseline.claim
        state = EpistemicState.SETTLED
        contradiction_events = []
        
        # 3. Sequential Belief Updating with Confirmation Rule
        # A contradiction only triggers EVOLVING if confirmed by >= min_confirming sources.
        for i in range(1, len(sorted_evidences)):
            new_item = sorted_evidences[i]
            
            if not new_item.claim:
                continue
                
            nli_res = await self.nli_engine.classify(current_belief, new_item.claim)
            label = nli_res.get("label", "NEUTRAL")
            conf = nli_res.get("confidence", 0.0)
            
            # Rule: contradiction_prob > 0.80 and reproducibility_score > 0.50
            if label == "CONTRADICTION" and conf > 0.80:
                if new_item.reproducibility_score > 0.50:
                    contradiction_events.append({
                        "pmid": new_item.pmid,
                        "year": new_item.metadata.year,
                        "reproducibility": new_item.reproducibility_score,
                        "claim": new_item.claim
                    })
                    
                    # Confirmation check
                    if len(contradiction_events) >= self.min_confirming:
                        if state == EpistemicState.SETTLED:
                            state = EpistemicState.EVOLVING
                        elif state == EpistemicState.EVOLVING:
                            state = EpistemicState.CONTESTED
                    
                    # Revise belief if new study is significantly higher quality
                    if new_item.reproducibility_score > baseline.reproducibility_score:
                        current_belief = new_item.claim
            
            elif label == "ENTAILMENT" and conf > 0.85:
                if state == EpistemicState.INSUFFICIENT:
                    state = EpistemicState.SETTLED

        return EpistemicResult(
            state=state,
            belief=baseline.reproducibility_score, # Simplified
            uncertainty=1.0 - baseline.reproducibility_score,
            conflict=len(contradiction_events) / len(evidences),
            temporal_shift_detected=(state in [EpistemicState.EVOLVING, EpistemicState.CONTESTED]),
            response_tier=ResponseTier.FULL if state == EpistemicState.SETTLED else ResponseTier.QUALIFIED,
            evidence_items=evidences,
            baseline_claim=baseline.claim,
            current_belief=current_belief,
            contradiction_events=contradiction_events
        )
