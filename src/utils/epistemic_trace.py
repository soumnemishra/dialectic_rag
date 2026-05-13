import json
import uuid
from datetime import datetime
from typing import Any, Dict, List

import logging

logger = logging.getLogger(__name__)

TRACE_VERSION = "1.0"


def _now_iso() -> str:
    return datetime.utcnow().isoformat()


def _summarize_value(value: Any, max_items: int = 5, max_str: int = 240) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        return value[:max_str]
    if isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, list):
        return {
            "count": len(value),
            "sample": [_summarize_value(v, max_items, max_str) for v in value[:max_items]],
        }
    if isinstance(value, dict):
        items = list(value.items())[:max_items]
        return {k: _summarize_value(v, max_items, max_str) for k, v in items}
    return str(value)[:max_str]


def _ensure_trace_context(state: Dict[str, Any]) -> Dict[str, str]:
    trace_id = str(state.get("trace_id") or "").strip()
    trace_created_at = str(state.get("trace_created_at") or "").strip()
    if not trace_id:
        trace_id = str(uuid.uuid4())
    if not trace_created_at:
        trace_created_at = _now_iso()
    return {"trace_id": trace_id, "trace_created_at": trace_created_at}


def build_trace_event(
    state: Dict[str, Any],
    section: str,
    event: str,
    data: Dict[str, Any] | None = None,
    node: str | None = None,
    influence: Dict[str, Any] | None = None,
    attach_context: bool = True,
) -> Dict[str, Any]:
    payload = {
        "section": section,
        "event": event,
    }
    if node:
        payload["node"] = node
    if data is not None:
        payload["data"] = _summarize_value(data)
    if influence is not None:
        payload["influence"] = _summarize_value(influence)

    if attach_context:
        context = _ensure_trace_context(state)
        payload["trace_version"] = TRACE_VERSION
        payload["trace_id"] = context["trace_id"]
        payload["ts"] = _now_iso()
    return payload


def build_trace_updates(state: Dict[str, Any], events: List[Dict[str, Any]]) -> Dict[str, Any]:
    context = _ensure_trace_context(state)
    for event in events:
        event.setdefault("trace_id", context["trace_id"])
        event.setdefault("trace_version", TRACE_VERSION)
        event.setdefault("ts", _now_iso())
        try:
            logger.info("TRACE %s", json.dumps(event, ensure_ascii=True))
        except Exception:
            logger.info("TRACE event emitted (json serialization failed)")
    return {
        "trace_id": context["trace_id"],
        "trace_created_at": context["trace_created_at"],
        "trace_events": events,
    }
# --- Observability Framework Constants ---

SNAPSHOT_FIELDS = {
    "clinical_intent": ["intent", "risk_level", "requires_disclaimer", "needs_guidelines"],
    "router": ["router_output"],
    "planner": ["plan", "plan_error"],
    "executor": ["step_output", "step_docs_ids", "step_notes"],
    "rag_direct": ["final_answer", "predicted_letter", "step_output", "step_docs_ids"],
    "evidence_polarity": ["evidence_polarity"],
    "evidence_decision": ["evidence_decision"],
    "temporal_conflict": ["tcs_score", "temporal_conflicts"],
    "rps_scoring": ["rps_scores"],
    "applicability_scoring": ["applicability_score"],
    "adversarial_retrieval": ["thesis_docs", "antithesis_docs"],
    "dialectical_synthesis": ["dialectic_synthesis", "final_answer", "predicted_letter"],
    "controversy_classifier": ["controversy_label"],
    "eup": ["eus_per_claim", "belief_intervals", "final_answer"],
    "decision_alignment": ["final_answer", "predicted_letter", "answer_source", "safety_flags", "switch_applied"],
    "safety_critic": ["final_answer", "safety_flags", "predicted_letter"],
}

GOVERNANCE_POLICIES = {
    "abstention": ["evidence_decision", "ds_uncertain", "ds_conflict"],
    "dialectical_activation": ["polarity", "tcs_score"],
    "synthesis_routing": ["avg_rps", "applicability_score", "polarity"],
    "belief_revision": ["overturned_pmids"],
    "answer_eligibility": ["ds_support", "calibrated_confidence"],
    "hypothesis_switching": ["polarity", "confidence"],
    "transparency": ["eus_value"],
    "safety": ["intent", "risk_level", "safety_flags"],
}


def assemble_structured_trace(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Assembles scattered trace events from state into a hierarchical causal trace.
    Implementation moved to trace_reporter.py to keep this utility lightweight.
    """
    from src.utils.trace_reporter import TraceReporter
    return TraceReporter.assemble(state)
