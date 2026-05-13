import logging
from typing import Dict, Any
from src.models.state import GraphState
from src.models.enums import EpistemicState
from src.epistemic.dempster_shafer import DempsterShaferIntegrator
from src.epistemic.epistemic_state_classifier import EpistemicStateClassifier
from src.epistemic.calibrated_abstention import CalibratedAbstention

logger = logging.getLogger(__name__)

async def uncertainty_propagation_node(state: GraphState) -> Dict[str, Any]:
    """
    Node to propagate uncertainty and classify epistemic state.
    Integrates D-S fusion, fuzzy classification, and abstention logic.
    """
    raw_pool = state.get("evidence_pool", [])
    temporal_result = state.get("temporal_result") # result from conflict_analysis
    
    # Instrumentation & Re-instantiation
    from src.models.schemas import EvidenceItem
    evidence_pool: List[EvidenceItem] = []
    
    logger.info(f"Uncertainty Propagation Input: {len(raw_pool)} items")
    for raw in raw_pool:
        if isinstance(raw, dict):
            item = EvidenceItem(**raw)
        else:
            item = raw
        evidence_pool.append(item)

    if not evidence_pool:
        return {"epistemic_result": None} 

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
        
        return {
            "epistemic_result": result,
            "abstention_rationale": rationale or "Error in uncertainty propagation."
        }
        
    except Exception as e:
        logger.error(f"Uncertainty propagation failed: {e}")
        return {"epistemic_result": None}
