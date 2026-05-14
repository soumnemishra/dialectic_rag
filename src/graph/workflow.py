import logging
from typing import Dict, Any, List, Optional
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field

from src.models.schemas import PICO, EvidenceItem, MassFunction, EpistemicState, ResponseTier, EpistemicResult
from src.models.state import GraphState
from src.nodes.pico_extraction import pico_extraction_node
from src.nodes.contrastive_retrieval import contrastive_retrieval_node
from src.nodes.epistemic_scoring import epistemic_scoring_node
from src.nodes.claim_clustering import claim_clustering_node
from src.nodes.conflict_analysis import conflict_analysis_node
from src.nodes.uncertainty_propagation import uncertainty_propagation_node
from src.nodes.response_generation import response_generation_node

logger = logging.getLogger(__name__)

def build_workflow(checkpointer=None):
    """Builds the compiled LangGraph workflow."""
    workflow = StateGraph(GraphState)

    # Add Nodes
    workflow.add_node("extract_pico", pico_extraction_node)
    workflow.add_node("contrastive_retrieval", contrastive_retrieval_node)
    workflow.add_node("epistemic_scoring", epistemic_scoring_node)
    workflow.add_node("claim_clustering", claim_clustering_node)
    workflow.add_node("conflict_analysis", conflict_analysis_node)
    workflow.add_node("uncertainty_propagation", uncertainty_propagation_node)
    workflow.add_node("response_generation", response_generation_node)

    def route_post_pico(state: GraphState) -> str:
        """Route to retrieval if PICO is valid, else skip to response generation."""
        pico = state.get("pico")
        # Handle both dict and PICO object cases
        if not pico:
            return "response_generation"
        
        # Check for minimal clinical components (e.g., population or intervention)
        if isinstance(pico, dict):
            pop = pico.get("population")
            inter = pico.get("intervention")
        else:
            pop = getattr(pico, "population", None)
            inter = getattr(pico, "intervention", None)
            
        if (not pop or pop == "unknown") and (not inter or inter == "unknown"):
            logger.warning("No clinical components found in PICO. Routing to early exit.")
            return "response_generation"
            
        return "contrastive_retrieval"

    # Add Edges (Sequential with conditional early exit)
    workflow.add_edge(START, "extract_pico")
    workflow.add_conditional_edges("extract_pico", route_post_pico)
    workflow.add_edge("contrastive_retrieval", "epistemic_scoring")
    workflow.add_edge("epistemic_scoring", "claim_clustering")
    workflow.add_edge("claim_clustering", "conflict_analysis")
    workflow.add_edge("conflict_analysis", "uncertainty_propagation")
    workflow.add_edge("uncertainty_propagation", "response_generation")
    workflow.add_edge("response_generation", END)

    return workflow.compile(checkpointer=checkpointer)
