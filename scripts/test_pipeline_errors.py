import logging
import sys
from typing import get_type_hints
from src.graph.workflow import build_workflow, GraphState
from src.nodes.pico_extraction import pico_extraction_node
from src.nodes.contrastive_retrieval import contrastive_retrieval_node
from src.nodes.epistemic_scoring import epistemic_scoring_node
from src.nodes.claim_clustering import claim_clustering_node
from src.nodes.conflict_analysis import conflict_analysis_node
from src.nodes.uncertainty_propagation import uncertainty_propagation_node
from src.nodes.response_generation import response_generation_node

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PipelineValidator")

def check_node_signatures():
    """Verify that all nodes use the correct GraphState type."""
    nodes = [
        ("pico_extraction", pico_extraction_node),
        ("contrastive_retrieval", contrastive_retrieval_node),
        ("epistemic_scoring", epistemic_scoring_node),
        ("claim_clustering", claim_clustering_node),
        ("conflict_analysis", conflict_analysis_node),
        ("uncertainty_propagation", uncertainty_propagation_node),
        ("response_generation", response_generation_node),
    ]
    
    passed = True
    for name, func in nodes:
        hints = get_type_hints(func)
        state_type = hints.get("state")
        if state_type != GraphState:
            logger.error(f"Node '{name}' has incorrect state type hint: {state_type} (expected {GraphState})")
            passed = False
        else:
            logger.info(f"Node '{name}' signature: OK")
    return passed

def main():
    logger.info("--- Starting Pipeline Validation ---")
    
    # 1. Check Node Signatures
    if not check_node_signatures():
        logger.error("Signature check FAILED.")
    else:
        logger.info("Signature check PASSED.")

    # 2. Try to Build Workflow
    try:
        app = build_workflow()
        logger.info("Workflow compilation: PASSED")
    except Exception as e:
        logger.error(f"Workflow compilation FAILED: {e}")
        sys.exit(1)

    logger.info("--- Validation Complete: READY FOR RUN ---")

if __name__ == "__main__":
    main()
