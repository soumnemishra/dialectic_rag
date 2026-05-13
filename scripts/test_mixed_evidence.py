import asyncio
import logging
from src.graph.workflow import build_workflow
from src.models.schemas import EvidenceItem, PICO
from src.models.enums import EvidenceStance, EpistemicState

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_mixed_evidence():
    print("\n--- RUNNING MIXED EVIDENCE TEST ---")
    app = build_workflow()
    
    # Mocking a state with 3 SUPPORT and 3 OPPOSE items
    # We'll bypass the extraction/clustering nodes by injecting them into the pool
    
    support_items = [
        EvidenceItem(
            pmid=f"100{i}",
            abstract="Study supporting the claim.",
            claim="Intervention reduces mortality.",
            stance=EvidenceStance.SUPPORT,
            reproducibility_score=0.8,
            applicability_score=0.8,
            nli_contradiction_prob=0.05
        ) for i in range(3)
    ]
    
    oppose_items = [
        EvidenceItem(
            pmid=f"200{i}",
            abstract="Study opposing the claim.",
            claim="Intervention does not reduce mortality.",
            stance=EvidenceStance.OPPOSE,
            reproducibility_score=0.8,
            applicability_score=0.8,
            nli_contradiction_prob=0.95
        ) for i in range(3)
    ]
    
    initial_state = {
        "original_question": "Does the intervention reduce mortality?",
        "evidence_pool": support_items + oppose_items,
        "pico": PICO(population="general", intervention="X", outcome="mortality")
    }
    
    # We want to run from uncertainty_propagation onwards
    # But for simplicity, we'll run the full graph and mock the earlier nodes
    # Or just run the specific node.
    
    # Let's run from uncertainty_propagation
    from src.nodes.uncertainty_propagation import uncertainty_propagation_node
    from src.nodes.response_generation import response_generation_node
    
    # Node 1: Uncertainty Propagation
    state_after_ds = await uncertainty_propagation_node(initial_state)
    initial_state.update(state_after_ds)
    
    res = initial_state["epistemic_result"]
    print(f"\nEpistemic State: {res.state}")
    print(f"Belief (True): {res.belief:.3f}")
    print(f"Conflict K: {res.conflict:.3f}")
    
    # Node 2: Response Generation
    final_state = await response_generation_node(initial_state)
    print("\n--- FINAL SYNTHESIS ---")
    print(final_state["candidate_answer"])

if __name__ == "__main__":
    asyncio.run(test_mixed_evidence())
