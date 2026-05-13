import logging
import time
from typing import Dict, Any, List, Tuple

from langgraph.graph import StateGraph, START, END

from src.state.state import GraphState
# Epistemic Reasoning Modules
from src.agents.clinical_intent       import clinical_intent_node
from src.agents.safety_critic         import safety_critic_node
from src.agents.rag_node              import rag_direct_node
from src.agents.evidence_polarity_agent  import evidence_polarity_node
from src.agents.evidence_governance    import evidence_governance_node
from src.agents.decision_alignment    import decision_alignment_node

# Epistemic Scoring Modules
from src.agents.temporal_conflict_node import temporal_conflict_node
from src.agents.rps_scoring_node import rps_scoring_node
from src.agents.applicability_node import applicability_scoring_node

# Dialectical Synthesis Modules
from src.agents.adversarial_retrieval_node import adversarial_retrieval_node
from src.agents.dialectical_synthesis_node import dialectical_synthesis_node
from src.agents.eup_node import eup_node
from src.agents.controversy_classifier_node import controversy_classifier_node

# Utilities
from src.agents.answer_utils import extract_final_answer_letter
from src.config import settings
from src.utils.epistemic_trace import build_trace_event, build_trace_updates
from src.core.thresholds import get_threshold_config

logger = logging.getLogger(__name__)

def build_graph():
    """
    Builds the Epistemically Aware Clinical RAG Pipeline.
    
    This pipeline implements the core scientific contribution:
    Modeling temporal conflict, methodological reliability, cohort applicability,
    contradictory findings, and epistemic uncertainty to generate safer and more
    transparent clinical recommendations.
    
    Architecture:
        START
          ↓
        clinical_intent         (Intent classification + risk assessment)
          ↓
        rag_direct              (Single retrieval call)
          ↓
        [PARALLEL EPISTEMIC SCORING]
        ├─ temporal_conflict    (Temporal Belief Revision: detect evidence shifts)
        ├─ rps_scoring          (Reproducibility Potential Score: methodological quality)
        └─ applicability_scoring (External Validity: cohort applicability)
          ↓
        epistemic_join          (Synchronization point)
          ↓
        evidence_polarity       (Evidence quality assessment: support/refute/uncertain)
          ↓
        evidence_governance     (Safety gate: abstain if insufficient)
          ↓
        controversy_classifier  (Epistemic status: settled/contested/evolving)
          ↓
        dialectic_gate          (Conditional trigger for contrastive retrieval)
          ├─ [IF contested/conflicting]
          │  adversarial_retrieval  (Retrieve counterarguments)
          │    ↓
          │  dialectical_synthesis  (Thesis-antithesis synthesis)
          └─ [ELSE: settled]
              (Skip contrastive retrieval)
          ↓
        eup                     (Epistemic Uncertainty Propagation: Dempster–Shafer fusion)
          ↓
        decision_alignment      (Align with clinical guidelines)
          ↓
        safety_critic           (Final safety validation)
          ↓
        END
    
    This modular design is interpretable, deterministic, and focused on the
    core research contribution of epistemic reasoning in clinical RAG.
    """

    workflow = StateGraph(GraphState)

    def _wrap_node(name, fn):
        async def _wrapped(state: GraphState):
            start = time.perf_counter()
            start_event = build_trace_event(state, section="node", event="start", node=name, attach_context=False)
            
            from src.utils.epistemic_trace import SNAPSHOT_FIELDS
            fields = SNAPSHOT_FIELDS.get(name, [])
            snap_pre = {f: state.get(f) for f in fields}
            
            try:
                result = await fn(state)
                if result is None: result = {}
                
                snap_post = {f: result.get(f, state.get(f)) for f in fields}
                diff = {f: {"pre": snap_pre.get(f), "post": snap_post.get(f)} for f in fields if snap_pre.get(f) != snap_post.get(f)}

                elapsed = (time.perf_counter() - start) * 1000.0
                if settings.LOG_NODE_TIMINGS:
                    logger.info("Node end | %s | %.1fms", name, elapsed)

                end_event = build_trace_event(
                    state,
                    section="node",
                    event="end",
                    node=name,
                    data={
                        "output_keys": sorted(list(result.keys())),
                        "duration_ms": round(elapsed, 1),
                        "snapshot_diff": diff if diff else None,
                    },
                    influence={"state_updates": sorted(list(result.keys()))},
                    attach_context=False,
                )
                trace_updates = build_trace_updates(state, [start_event, end_event])

                merged = dict(result)
                merged["trace_events"] = merged.get("trace_events", []) + trace_updates.get("trace_events", [])
                if "trace_id" not in merged: merged["trace_id"] = trace_updates.get("trace_id")
                if "trace_created_at" not in merged: merged["trace_created_at"] = trace_updates.get("trace_created_at")
                return merged
            except Exception as e:
                logger.error("Node error | %s | %s", name, e, exc_info=True)
                error_event = build_trace_event(state, section="node", event="error", node=name, data={"error": str(e)}, attach_context=False)
                build_trace_updates(state, [start_event, error_event])
                raise
        return _wrapped

    # --- Nodes ---
    workflow.add_node("clinical_intent",        _wrap_node("clinical_intent", clinical_intent_node))
    workflow.add_node("rag_direct",             _wrap_node("rag_direct", rag_direct_node))
    workflow.add_node("temporal_conflict",      _wrap_node("temporal_conflict", temporal_conflict_node))
    workflow.add_node("rps_scoring",            _wrap_node("rps_scoring", rps_scoring_node))
    workflow.add_node("applicability_scoring",  _wrap_node("applicability_scoring", applicability_scoring_node))
    
    async def epistemic_join_node(state: GraphState) -> Dict[str, Any]:
        return {}
    workflow.add_node("epistemic_join", _wrap_node("epistemic_join", epistemic_join_node))
    
    workflow.add_node("evidence_polarity",      _wrap_node("evidence_polarity", evidence_polarity_node))
    workflow.add_node("evidence_governance",    _wrap_node("evidence_governance", evidence_governance_node))
    workflow.add_node("controversy_classifier", _wrap_node("controversy_classifier", controversy_classifier_node))
    
    async def dialectic_gate_node(state: GraphState) -> Dict[str, Any]:
        return {} # Logic handled in conditional edges
    workflow.add_node("dialectic_gate",         _wrap_node("dialectic_gate", dialectic_gate_node))
    
    workflow.add_node("adversarial_retrieval",  _wrap_node("adversarial_retrieval", adversarial_retrieval_node))
    workflow.add_node("dialectical_synthesis",  _wrap_node("dialectical_synthesis", dialectical_synthesis_node))
    workflow.add_node("eup",                    _wrap_node("eup", eup_node))
    workflow.add_node("decision_alignment",     _wrap_node("decision_alignment", decision_alignment_node))
    workflow.add_node("safety_critic",          _wrap_node("safety_critic", safety_critic_node))

    # --- Edges ---
    workflow.add_edge(START, "clinical_intent")
    workflow.add_edge("clinical_intent", "rag_direct")
    
    # Parallel scoring
    workflow.add_edge("rag_direct", "temporal_conflict")
    workflow.add_edge("rag_direct", "rps_scoring")
    workflow.add_edge("rag_direct", "applicability_scoring")
    
    workflow.add_edge("temporal_conflict", "epistemic_join")
    workflow.add_edge("rps_scoring", "epistemic_join")
    workflow.add_edge("applicability_scoring", "epistemic_join")
    
    workflow.add_edge("epistemic_join", "evidence_polarity")
    workflow.add_edge("evidence_polarity", "evidence_governance")

    def route_after_governance(state: GraphState) -> str:
        if state.get("governance_decision") == "abstain":
            return "safety_critic"
        return "controversy_classifier"

    workflow.add_conditional_edges(
        "evidence_governance",
        route_after_governance,
        {"abstain": "safety_critic", "controversy_classifier": "controversy_classifier"}
    )

    workflow.add_edge("controversy_classifier", "dialectic_gate")

    def route_dialectic(state: GraphState) -> str:
        # Simplified trigger logic for dialectical retrieval
        controversy = str(state.get("controversy_label", "")).upper()
        tcs = float(state.get("tcs_score", 0.0))
        polarity = state.get("evidence_polarity", {}).get("polarity", "insufficient")
        
        thresh = get_threshold_config()
        
        # Trigger adversarial if contested/evolving or specific risk markers met
        if (controversy in {"CONTESTED", "EVOLVING"} or 
            tcs > thresh.temporal_conflict.tcs_trigger or 
            polarity == "insufficient"):
            logger.info("Dialectic Gate: Routing to adversarial retrieval.")
            return "adversarial_retrieval"
        
        logger.info("Dialectic Gate: Routing to EUP (settled).")
        return "eup"

    workflow.add_conditional_edges(
        "dialectic_gate",
        route_dialectic,
        {"adversarial_retrieval": "adversarial_retrieval", "eup": "eup"}
    )

    workflow.add_edge("adversarial_retrieval", "dialectical_synthesis")
    workflow.add_edge("dialectical_synthesis", "eup")
    
    workflow.add_edge("eup", "decision_alignment")
    workflow.add_edge("decision_alignment", "safety_critic")
    workflow.add_edge("safety_critic", END)

    return workflow.compile()