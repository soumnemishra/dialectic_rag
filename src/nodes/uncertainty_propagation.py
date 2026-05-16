from __future__ import annotations
import logging
from typing import Any
from src.models.state import GraphState
from src.models.schemas import EvidenceItem
from src.models.enums import EpistemicState
from src.epistemic.dempster_shafer import DempsterShaferIntegrator
from src.epistemic.epistemic_state_classifier import EpistemicStateClassifier
from src.epistemic.calibrated_abstention import CalibratedAbstention
from src.config import epistemic_settings

logger = logging.getLogger(__name__)


def _stance_value(item: EvidenceItem) -> str:
    return str(getattr(item.stance, "value", item.stance or "NEUTRAL"))

async def uncertainty_propagation_node(state: GraphState) -> dict[str, Any]:
    """
    Node to propagate uncertainty and classify epistemic state.
    Integrates D-S fusion, fuzzy classification, and abstention logic.
    """
    raw_pool = state.get("evidence_pool", [])
    temporal_result = state.get("temporal_result") # result from conflict_analysis
    
    # Instrumentation & Re-instantiation
    from src.models.schemas import EvidenceItem
    evidence_pool: list[EvidenceItem] = []
    
    logger.info(f"Uncertainty Propagation Input: {len(raw_pool)} items")
    for raw in raw_pool:
        if isinstance(raw, dict):
            item = EvidenceItem(**raw)
        else:
            item = raw
        evidence_pool.append(item)

    if not evidence_pool:
        from src.models.schemas import EpistemicResult, ResponseTier
        fallback = EpistemicResult(
            state=EpistemicState.INSUFFICIENT,
            belief=0.0,
            uncertainty=1.0,
            conflict=0.0,
            temporal_shift_detected=False,
            response_tier=ResponseTier.ABSTAIN,
            evidence_items=[]
        )
        return {
            "epistemic_result": fallback,
            "abstention_rationale": "No evidence items survived the epistemic gating process."
        }

    integrator = DempsterShaferIntegrator()
    classifier = EpistemicStateClassifier()
    abstainer = CalibratedAbstention()

    try:
        # 1. Fuse evidence pool
        logger.info(f"Fusing evidence pool: {len(evidence_pool)} items")
        for item in evidence_pool:
            m = integrator.assign_mass(item)
            logger.info(f"Item {item.pmid} Mass: True={m.belief_true:.3f}, False={m.belief_false:.3f}, U={m.uncertainty:.3f} (Stance={item.stance})")
            
        combined_mass, max_k = integrator.fuse_pool(evidence_pool)
        
        # 2. Get pignistic belief (P(true))
        belief = integrator.pignistic_probability(combined_mass)
        uncertainty = combined_mass.uncertainty
        conflict = max_k
        
        # 3. Detect temporal shift (from previous node)
        temporal_shift = False
        if temporal_result and temporal_result.state == EpistemicState.EVOLVING:
            temporal_shift = True
            
        # 4. Classify Epistemic State (Fuzzy Logic)
        state_class = classifier.classify(
            belief=belief,
            uncertainty=uncertainty,
            conflict=conflict,
            temporal_shift_detected=temporal_shift,
            belief_true=combined_mass.belief_true,
            belief_false=combined_mass.belief_false
        )
        
        # 5. Calibrated Abstention Logic
        tier, rationale = abstainer.should_abstain(
            epistemic_state=state_class,
            belief=belief,
            uncertainty=uncertainty,
            conflict_K=conflict
        )
        
        from src.models.schemas import EpistemicResult, ResponseTier
        try:
            result = EpistemicResult(
                state=state_class,
                belief=belief,
                uncertainty=uncertainty,
                conflict=conflict,
                temporal_shift_detected=temporal_shift,
                response_tier=tier,
                evidence_items=evidence_pool,
                baseline_claim=temporal_result.baseline_claim if temporal_result else None,
                current_belief=temporal_result.current_belief if temporal_result else None,
                contradiction_events=temporal_result.contradiction_events if temporal_result else []
            )
        except Exception as e:
            logger.error(f"Failed to construct EpistemicResult: {e}")
            # Fallback to a safe ABSTAIN state
            result = EpistemicResult(
                state=EpistemicState.INSUFFICIENT,
                belief=0.0,
                uncertainty=1.0,
                conflict=0.0,
                temporal_shift_detected=False,
                response_tier=ResponseTier.ABSTAIN,
                evidence_items=evidence_pool
            )
        
        # Add trace event for D-S Fusion (requested in audit)
        trace_event = {
            "node": "uncertainty_propagation",
            "section": "dempster_shafer_combination",
            "equation_alignment": {
                "equations": {
                    "evidence_weight": "Eq. (6) w=RPS*A",
                    "mass_assignment": "Eq. (7) m(T/F)=gamma*w; m(Theta)=1-m(T)-m(F)",
                    "conflict": "Eq. (8)-(9) K and K_global",
                    "decision_probability": "Eq. (10) BetP(T)=m(T)+0.5*m(Theta)",
                },
                "yaml_sections": {
                    "ds": epistemic_settings.get("ds", {}),
                    "evidence_gating": epistemic_settings.get("evidence_gating", {}),
                    "states": epistemic_settings.get("states", {}),
                    "abstention": epistemic_settings.get("abstention", {}),
                },
            },
            "output": {
                "final_belief_masses": {
                    "belief_true": round(combined_mass.belief_true, 4),
                    "belief_false": round(combined_mass.belief_false, 4),
                    "uncertainty": round(combined_mass.uncertainty, 4)
                },
                "conflict_mass": round(conflict, 4),
                "pignistic_belief": round(belief, 4),
                "epistemic_state": state_class.value,
                "decision": tier.value,
                "rationale": rationale,
                "stance_counts": {
                    "SUPPORT": sum(1 for item in evidence_pool if _stance_value(item) == "SUPPORT"),
                    "OPPOSE": sum(1 for item in evidence_pool if _stance_value(item) == "OPPOSE"),
                    "REFINE": sum(1 for item in evidence_pool if _stance_value(item) == "REFINE"),
                    "NEUTRAL": sum(1 for item in evidence_pool if _stance_value(item) == "NEUTRAL"),
                }
            }
        }
        
        return {
            "epistemic_result": result,
            "abstention_rationale": rationale or "Error in uncertainty propagation.",
            "trace_events": [trace_event]
        }
        
    except Exception as e:
        logger.error(f"Uncertainty propagation failed: {e}")
        from src.models.schemas import EpistemicResult, ResponseTier
        fallback = EpistemicResult(
            state=EpistemicState.INSUFFICIENT,
            belief=0.0,
            uncertainty=1.0,
            conflict=0.0,
            temporal_shift_detected=False,
            response_tier=ResponseTier.ABSTAIN,
            evidence_items=evidence_pool
        )
        return {
            "epistemic_result": fallback,
            "abstention_rationale": f"System error during belief fusion: {str(e)}"
        }
