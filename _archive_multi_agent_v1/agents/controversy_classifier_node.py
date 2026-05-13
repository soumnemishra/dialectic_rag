import logging
from typing import Any, Dict, List
import os

from src.state.state import GraphState
from src.utils.epistemic_trace import build_trace_event, build_trace_updates

logger = logging.getLogger(__name__)


def _step_consensus(step_output: List[Dict[str, Any]]) -> tuple[float, Dict[str, int]]:
    counts: Dict[str, int] = {}
    definitive = 0
    for output in step_output or []:
        if not isinstance(output, dict):
            continue
        if str(output.get("retrieval_status", "")).upper() == "MISSING":
            continue
        answer = str(output.get("predicted_letter") or output.get("answer") or "").strip().upper()
        if answer in {"", "UNKNOWN"}:
            continue
        definitive += 1
        counts[answer] = counts.get(answer, 0) + 1

    if definitive <= 0 or not counts:
        return 0.0, counts

    return max(counts.values()) / definitive, counts


def compute_controversy_label_from_scores(
    tcs: float,
    polarity: str,
    confidence: float,
    rps_avg: float,
    applicability_score: float,
    has_conflict: bool = False,
    antithesis_count: int = 0,
    step_consensus: float = 1.0,
    split_counts: Dict[str, int] | None = None,
) -> str:
    """Deterministically classify controversy from epistemic scores."""
    split_counts = split_counts or {}
    split_has_two_sides = sum(1 for count in split_counts.values() if count > 0) >= 2 and sum(1 for count in split_counts.values() if count >= 2) >= 2

    if tcs > 0.10 or has_conflict or antithesis_count > 0 or split_has_two_sides:
        return "CONTESTED"

    if rps_avg < 0.20 and step_consensus < 0.50:
        return "CONTESTED"

    if tcs > 0.0:
        return "EVOLVING"

    if polarity in ["support", "refute"] and confidence >= 0.8 and rps_avg >= 0.35 and applicability_score >= 0.6:
        logger.info(f"Overriding to SETTLED based on strong {polarity} and zero conflict indicators.")
        return "SETTLED"

    if rps_avg < 0.35 and step_consensus >= 0.50:
        logger.info(f"Defaulting to SETTLED due to zero conflict indicators despite borderline scores.")
        return "SETTLED"

    return "EVOLVING"


async def controversy_classifier_node(state: GraphState) -> Dict[str, Any]:
    """
        Reads:  state["tcs_score"], state["evidence_polarity"], state["rps_scores"],
            state["dialectic_synthesis"], state["applicability_score"]
    Writes: state["controversy_label"]

    The dialectical synthesis label is treated as advisory. This node
    deterministically computes the final controversy label so the field
    is always available for evaluation metadata.
    """
    tcs_raw = state.get("tcs_score")
    tcs = float(tcs_raw if tcs_raw is not None else 0.0)
    polarity_data = state.get("evidence_polarity", {})
    synthesis = state.get("dialectic_synthesis", {})
    rps_scores: List[Dict[str, Any]] = state.get("rps_scores", [])
    step_output: List[Dict[str, Any]] = state.get("step_output", [])

    polarity = "insufficient"
    confidence = 0.0
    if isinstance(polarity_data, dict):
        polarity = str(polarity_data.get("polarity", "insufficient")).lower()
        confidence = float(polarity_data.get("confidence", 0.0))

    has_conflict = False
    antithesis_count = 0
    llm_label = ""

    if isinstance(synthesis, dict):
        has_conflict = bool(synthesis.get("has_conflict", False))
        antithesis_count = int(synthesis.get("antithesis_count", 0) or 0)
        llm_label = str(synthesis.get("controversy_label", "")).upper()

    # Treat missing/empty RPS as neutral reproducibility (0.5) — defensive parsing
    valid_rps: List[float] = []
    for s in rps_scores:
        val = s.get("final_score")
        if val is None:
            val = s.get("rps_score", s.get("rps", 0.5))
        if val is None:
            continue
        try:
            valid_rps.append(float(val))
        except (TypeError, ValueError):
            continue
    avg_rps = sum(valid_rps) / len(valid_rps) if valid_rps else 0.5
    applicability_score = float(state.get("applicability_score", 0.0))
    step_consensus, split_counts = _step_consensus(step_output)

    # Capture thresholds for tracing
    thresholds = {
        "settled_rps_min": float(os.getenv("MRAGE_SETTLED_RPS_THRESHOLD", "0.35")),
        "settled_app_min": float(os.getenv("MRAGE_SETTLED_APPLICABILITY_THRESHOLD", "0.6")),
        "settled_conf_min": float(os.getenv("MRAGE_SETTLED_CONFIDENCE", "0.8")),
        "settled_tcs_max": float(os.getenv("MRAGE_SETTLED_TCS_MAX", "0.2")),
        "evolving_tcs_min": float(os.getenv("MRAGE_EVOLVING_TCS_MIN", "0.4")),
        "contested_conf_max": float(os.getenv("MRAGE_CONTESTED_CONFIDENCE_MAX", "0.6")),
        "emerging_rps_max": float(os.getenv("MRAGE_EMERGING_RPS_MAX", "0.1")),
    }

    label = compute_controversy_label_from_scores(
        tcs=tcs,
        polarity=polarity,
        confidence=confidence,
        rps_avg=avg_rps,
        applicability_score=applicability_score,
        has_conflict=has_conflict,
        antithesis_count=antithesis_count,
        step_consensus=step_consensus,
        split_counts=split_counts,
    )

    logger.info(
        "controversy_classifier: tcs=%.3f polarity=%s confidence=%.2f avg_rps=%.3f step_consensus=%.3f has_conflict=%s antithesis_count=%s llm_label=%s final_label=%s",
        tcs,
        polarity,
        confidence,
        avg_rps,
        step_consensus,
        has_conflict,
        antithesis_count,
        llm_label or "N/A",
        label,
    )
    payload = {"controversy_label": label}
    trace_event = build_trace_event(
        state,
        section="controversy_analysis",
        event="controversy_classification",
        node="controversy_classifier",
        data={
            "inputs": {
                "tcs_score": round(tcs, 3),
                "polarity": polarity,
                "confidence": round(confidence, 3),
                "avg_rps": round(avg_rps, 3),
                "step_consensus": round(step_consensus, 3),
                "applicability_score": round(applicability_score, 3),
                "has_conflict": has_conflict,
                "antithesis_count": antithesis_count,
                "llm_label": llm_label or "N/A",
            },
            "thresholds": thresholds,
            "threshold_crossings": {
                "high_temporal_conflict": tcs >= thresholds["evolving_tcs_min"],
                "dialectical_conflict": has_conflict or antithesis_count > 0,
                "low_confidence": (polarity in ["support", "refute"]) and confidence < thresholds["contested_conf_max"],
                "low_reproducibility": avg_rps < thresholds["emerging_rps_max"],
                "settled_consensus": (polarity in ["support", "refute"] and confidence >= thresholds["settled_conf_min"] and avg_rps >= thresholds["settled_rps_min"] and applicability_score >= thresholds["settled_app_min"]),
            },
            "final_label": label,
        },
        influence={"state_updates": ["controversy_label"]},
        attach_context=False,
    )

    trace_updates = build_trace_updates(state, [trace_event])
    payload["trace_events"] = payload.get("trace_events", []) + trace_updates.get("trace_events", [])
    if "trace_id" not in payload:
        payload["trace_id"] = trace_updates.get("trace_id")
    if "trace_created_at" not in payload:
        payload["trace_created_at"] = trace_updates.get("trace_created_at")
    return payload
