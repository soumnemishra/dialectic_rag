import logging
import os
import re
from typing import Any, Dict, List

from src.config import settings

from src.state.state import GraphState
from src.epistemic.dempster_shafer import (
    build_epistemic_masses,
    fuse_masses,
    build_mass as _build_mass,
    combine_masses as _combine_masses,
    controversy_mass as _controversy_mass,
    dialectic_mass as _dialectic_mass,
    temporal_mass as _temporal_mass,
)
from src.utils.epistemic_trace import build_trace_event, build_trace_updates

logger = logging.getLogger(__name__)


EPISTEMIC_DISCOUNT_BELIEF_CAP = 0.75
YAGER_CONFLICT_THRESHOLD = 0.85
HARD_ABSTENTION_BASE = 0.80


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value if value is not None and value != "" else default)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value if value is not None and value != "" else default)
    except (TypeError, ValueError):
        return default


def _build_mass_placeholder():
    # Backward-compat shim for legacy tests.
    return _build_mass(0.0, 0.0)


def _eus_label(eus: float) -> str:
    if eus <= 0.20:
        return "WELL-SUPPORTED"
    if eus <= 0.35:
        return "MODERATE CONFIDENCE"
    if eus <= 0.55:
        return "WARRANTED BUT CONTESTED"
    if eus <= 0.70:
        return "HIGH UNCERTAINTY"
    return "INSUFFICIENT EVIDENCE"


def _step_signal_counts(step_output: List[Dict[str, Any]]) -> tuple[float, float, float, Dict[str, int]]:
    definitive = 0
    unknown_or_missing = 0
    zero_docs = 0
    modal_counts: Dict[str, int] = {}

    for output in step_output or []:
        if not isinstance(output, dict):
            continue
        retrieval_status = str(output.get("retrieval_status", "")).upper()
        if retrieval_status == "MISSING":
            unknown_or_missing += 1
            zero_docs += 1
            continue

        doc_ids = output.get("doc_ids")
        if isinstance(doc_ids, list):
            if len(doc_ids) == 0:
                zero_docs += 1
        elif doc_ids in (None, ""):
            zero_docs += 1

        answer = str(output.get("predicted_letter") or output.get("answer") or "").strip().upper()
        if answer in {"", "UNKNOWN"}:
            unknown_or_missing += 1
            continue
        definitive += 1
        modal_counts[answer] = modal_counts.get(answer, 0) + 1

    total = max(len(step_output or []), 1)
    unknown_fraction = unknown_or_missing / total
    zero_doc_fraction = zero_docs / total
    consensus = 0.0
    if definitive > 0 and modal_counts:
        consensus = max(modal_counts.values()) / definitive
    return consensus, unknown_fraction, zero_doc_fraction, modal_counts


def _annotate_answer(answer: str, eus_value: float) -> str:
    if not answer:
        return answer
    label = _eus_label(eus_value)
    return f"{answer}\n\n**Epistemic Uncertainty Score (EUS):** {eus_value:.2f} [{label}]"


def get_adaptive_abstention_threshold(
    applicability_score: float, polarity: str, controversy_label: str
) -> float:
    """
    Settled + Strong Support + High Applicability = very high threshold.
    """
    base = HARD_ABSTENTION_BASE

    if polarity == "strong_support":
        base += 0.10  # 0.90
    if str(controversy_label).upper() == "SETTLED":
        base += 0.05  # 0.95
    if applicability_score >= 0.90:
        base += 0.03  # 0.98

    return min(base, 0.98)


def _extract_claims(answer: str, max_claims: int) -> List[str]:
    if not answer or max_claims <= 0:
        return []

    claims: List[str] = []
    for line in str(answer).splitlines():
        trimmed = line.strip()
        if not trimmed:
            continue
        if re.match(r"^(?:[-*]|\d+[\).])\s+", trimmed):
            claim = re.sub(r"^(?:[-*]|\d+[\).])\s+", "", trimmed).strip()
            if claim:
                claims.append(claim)

    if not claims:
        sentences = re.split(r"(?<=[.!?])\s+", str(answer).strip())
        for sentence in sentences:
            cleaned = sentence.strip()
            if len(cleaned) < 12:
                continue
            claims.append(cleaned)

    deduped: List[str] = []
    seen: set[str] = set()
    for claim in claims:
        if claim in seen:
            continue
        seen.add(claim)
        deduped.append(claim)
        if len(deduped) >= max_claims:
            break

    return deduped


def _derive_answer_confidence_label(state: GraphState) -> str:
    """Best-effort answer confidence label for EUS calibration."""
    step_output = state.get("step_output") or []
    if isinstance(step_output, list) and step_output:
        last = step_output[-1]
        if isinstance(last, dict):
            raw = last.get("confidence")
            if isinstance(raw, str) and raw.strip():
                value = raw.strip().upper()
                if value in {"HIGH", "MEDIUM", "LOW"}:
                    return value

            raw_prob = last.get("epistemic_confidence")
            prob = _safe_float(raw_prob, -1.0)
            if 0.0 <= prob <= 1.0:
                if prob >= 0.70:
                    return "HIGH"
                if prob >= 0.50:
                    return "MEDIUM"
                return "LOW"

    # Fall back to evidence polarity confidence when explicit answer confidence is absent.
    pol = state.get("evidence_polarity", {})
    if isinstance(pol, dict):
        prob = _safe_float(pol.get("confidence"), -1.0)
        if 0.0 <= prob <= 1.0:
            if prob >= 0.75:
                return "HIGH"
            if prob >= 0.55:
                return "MEDIUM"
            return "LOW"

    return "LOW"


async def eup_node(state: GraphState) -> Dict[str, Any]:
    """
    Fuse epistemic evidence channels with Dempster–Shafer theory and emit
    a calibrated uncertainty estimate for the final clinical answer.
    """
    logger.info(f"EPISTEMIC_MODE inside eup_node = {settings.EPISTEMIC_MODE}")
    tcs_score = _safe_float(state.get("tcs_score"), 0.0)
    applicability_raw = state.get("applicability_score")
    applicability = _safe_float(applicability_raw, 0.0)
    rps_scores: List[Dict[str, Any]] = state.get("rps_scores") or []
    if not isinstance(rps_scores, list):
        rps_scores = []
    synthesis = state.get("dialectic_synthesis") or {}
    if not isinstance(synthesis, dict):
        synthesis = {}
    final_answer = str(state.get("final_answer") or "")
    controversy_label = str(state.get("controversy_label") or "")

    try:
        valid_rps_scores = []
        for r in rps_scores:
            val = r.get("final_score") if r.get("final_score") is not None else r.get("rps_score")
            if val is not None:
                try:
                    valid_rps_scores.append(float(val))
                except (TypeError, ValueError):
                    pass
        avg_rps = sum(valid_rps_scores) / len(valid_rps_scores) if valid_rps_scores else None
        thesis_count = _safe_int(synthesis.get("thesis_count"), 0)
        antithesis_count = _safe_int(synthesis.get("antithesis_count"), 0)
        total = max(thesis_count + antithesis_count, 1)
        conflict_ratio = antithesis_count / total

        evidence_polarity = state.get("evidence_polarity", {})
        polarity = evidence_polarity.get("polarity", "insufficient") if isinstance(evidence_polarity, dict) else "insufficient"
        polarity_conf = _safe_float(
            evidence_polarity.get("confidence") if isinstance(evidence_polarity, dict) else 0.0,
            0.0,
        )

        masses = build_epistemic_masses(
            tcs_score=tcs_score if tcs_score and float(tcs_score) > 0.0 else None,
            avg_rps=avg_rps,
            applicability=applicability if applicability_raw is not None else None,
            controversy_label=controversy_label,
            conflict_ratio=conflict_ratio,
            polarity=polarity,
            polarity_confidence=polarity_conf,
        )

        from src.epistemic.dempster_shafer import fuse_masses_with_lineage
        combined, ds_conflict, lineage = fuse_masses_with_lineage(masses)

        logger.info("Raw DS combined mass: %s", combined)

        # DS-derived belief, plausibility, and epistemic uncertainty.
        belief = float(combined.get("SUPPORT", 0.0))
        refute = float(combined.get("REFUTE", 0.0))
        residual_uncertain = float(combined.get("UNCERTAIN", 0.0))
        ds_conflict = float(ds_conflict or 0.0)
        belief = min(max(belief, 0.0), 1.0)
        refute = min(max(refute, 0.0), 1.0)
        residual_uncertain = min(max(residual_uncertain, 0.0), 1.0)
        ds_conflict = min(max(ds_conflict, 0.0), 1.0)

        step_output = state.get("step_output") or []
        step_consensus, unknown_fraction, zero_doc_fraction, modal_counts = _step_signal_counts(step_output if isinstance(step_output, list) else [])

        plausibility = min(1.0, max(0.0, belief + residual_uncertain))
        eus_value = residual_uncertain + 0.5 * ds_conflict
        eus_value = min(max(eus_value, 0.0), 1.0)

        dial_meta = state.get("dialectical_metadata", {})
        opposing_evidence_found = bool(state.get("opposing_evidence_found", False))
        if (
            isinstance(dial_meta, dict)
            and dial_meta.get("dialectical_search_executed")
            and (dial_meta.get("dialectical_zero_hit") or not opposing_evidence_found)
        ):
            eus_value = max(eus_value, 0.70)

        label_text = _eus_label(eus_value)

        # final numeric rounding
        belief = round(belief, 3)
        refute = round(refute, 3)
        residual_uncertain = round(residual_uncertain, 3)
        plausibility = round(plausibility, 3)
        eus_value = round(eus_value, 3)
        ds_conflict = round(ds_conflict, 3)

        # Adaptive abstention remains in force, but now it is driven purely by
        # the DS-derived EUS rather than any auxiliary heuristic score.
        abstention_threshold = get_adaptive_abstention_threshold(
            applicability, polarity, controversy_label
        )
        
        if eus_value >= abstention_threshold:
            logger.warning(
                f"Hard Abstention Triggered | EUS={eus_value:.2f} >= Threshold={abstention_threshold:.2f} "
                f"Applicability={applicability:.2f} Polarity={polarity}"
            )
            abstention_msg = (
                "**INSUFFICIENT EVIDENCE**: The retrieved literature does not provide a scientifically "
                "robust basis for a clinical recommendation. "
                f"(Reason: {'High uncertainty' if eus_value >= abstention_threshold else 'Low cohort applicability'})"
            )
            final_answer = abstention_msg
            label_text = "INSUFFICIENT EVIDENCE"

    except Exception as exc:
        logger.warning("EUP computation fallback applied: %s", exc)
        belief = 0.3
        plausibility = 0.8
        eus_value = 0.5
        ds_conflict = 0.0
        refute = 0.0
        residual_uncertain = 0.5
        lineage = []

    eus_per_claim = {"global": eus_value}

    belief_intervals = {
        "global": {
            "belief": round(belief, 3),
            "plausibility": round(plausibility, 3),
            "conflict": round(ds_conflict, 3),
        }
    }

    per_claim_enabled = os.getenv("MRAGE_EUS_PER_CLAIM", "0").strip().lower() in {"1", "true", "yes"}
    if per_claim_enabled:
        try:
            max_claims = int(os.getenv("MRAGE_EUS_MAX_CLAIMS", "5"))
        except ValueError:
            max_claims = 5
        claims = _extract_claims(final_answer, max_claims)
        for claim in claims:
            eus_per_claim[claim] = eus_value
            belief_intervals[claim] = {
                "belief": round(belief, 3),
                "plausibility": round(plausibility, 3),
            }

    # The label is derived from the DS uncertainty score.
    if label_text is None:
        label_text = _eus_label(eus_value)
    annotated_answer = (
        f"{final_answer}\n\n**Epistemic Uncertainty Score (EUS):** {eus_value:.2f} [{label_text}]"
    )

    # Prefix the answer when the DS uncertainty estimate is high.
    if settings.EPISTEMIC_MODE and eus_value >= 0.25:
        epistemic_prefix = (
            "\u26a0\ufe0f **EPISTEMIC NOTICE**: This answer carries significant uncertainty "
            f"(EUS={eus_value:.2f}, label={_eus_label(eus_value)}). "
            "Multiple evidence channels indicate limited confidence. "
            "The following answer should be interpreted with caution.\n\n"
        )
        annotated_answer = epistemic_prefix + annotated_answer

    logger.info(
        "EUP complete: claims=%s eus=%.3f belief=%.3f plausibility=%.3f",
        max(len(eus_per_claim) - 1, 0),
        eus_value,
        belief,
        plausibility,
    )

    trace_event = build_trace_event(
        state,
        section="ds_fusion",
        event="eup_fusion",
        node="eup",
        data={
            "inputs": {
                "tcs_score": round(tcs_score, 3),
                "applicability_score": round(applicability, 3),
                "avg_rps": round(avg_rps, 3) if avg_rps is not None else None,
                "controversy_label": controversy_label,
                "conflict_ratio": round(conflict_ratio, 3),
                "polarity": polarity,
                "polarity_confidence": round(polarity_conf, 3),
            },
            "masses": {
                "support": round(float(belief), 3),
                "refute": round(float(refute), 3),
                "uncertain": round(float(residual_uncertain), 3), # Raw DS mass, not EUS
                "conflict": round(float(ds_conflict), 3),
                "plausibility": round(float(plausibility), 3),
                "eus": round(float(eus_value), 3),
            },
            "fusion_lineage": lineage,
            "support_origin_details": {
                "direct_evidence": {
                    "polarity": polarity,
                    "confidence": round(polarity_conf, 3),
                },
                "applicability": round(applicability, 3),
                "rps_avg": round(avg_rps, 3) if avg_rps is not None else None,
                "tcs_score": round(tcs_score, 3),
                "fusion_model": "dempster_shafer",
            },
        },
        influence={"state_updates": ["eus_per_claim", "belief_intervals"]},
        attach_context=False,
    )

    trace_updates = build_trace_updates(state, [trace_event])
    payload = {
        "eus_per_claim": eus_per_claim,
        "belief_intervals": belief_intervals,
        "final_answer": annotated_answer,
    }
    payload["trace_events"] = payload.get("trace_events", []) + trace_updates.get("trace_events", [])
    if "trace_id" not in payload:
        payload["trace_id"] = trace_updates.get("trace_id")
    if "trace_created_at" not in payload:
        payload["trace_created_at"] = trace_updates.get("trace_created_at")
    return payload
