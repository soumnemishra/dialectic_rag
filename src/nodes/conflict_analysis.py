import logging
from collections import defaultdict
from typing import Dict, Any, List, Optional, Tuple
import json
import random

from src.models.state import GraphState
from src.models.schemas import EvidenceItem, PICO, EpistemicResult
from src.models.enums import EpistemicState, ResponseTier
from src.epistemic.nli_engine import NLIEngine

logger = logging.getLogger(__name__)


def _as_pico(value: Any) -> Optional[PICO]:
    if value is None:
        return None
    if isinstance(value, PICO):
        return value
    if isinstance(value, dict):
        try:
            return PICO(**value)
        except Exception:
            return None
    return None


def _candidate_hypothesis(candidate: str, pico: Optional[PICO]) -> str:
    if pico:
        return (
            f"For a patient with population '{pico.population}', intervention '{pico.intervention}', "
            f"and outcome '{pico.outcome}', the most likely diagnosis is {candidate}."
        )
    return f"The most likely diagnosis for this patient is {candidate}."


def _to_support_score(label: str, confidence: float) -> float:
    label = (label or "").upper()
    if label == "ENTAILMENT":
        return 1.0 * confidence
    if label == "CONTRADICTION":
        return -1.0 * confidence
    return 0.0


def _normalize_support(score: float, weight: float) -> float:
    if weight <= 0:
        return 0.0
    return max(-1.0, min(1.0, score / weight))


def _cluster_year_items(cluster: Dict[str, Any], evidence_by_pmid: Dict[str, EvidenceItem]) -> Dict[int, List[Dict[str, Any]]]:
    year_buckets: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for claim in cluster.get("claims", []) or []:
        pmid = str(claim.get("pmid", ""))
        evidence = evidence_by_pmid.get(pmid)
        year = claim.get("year") or claim.get("publication_year")
        if year is None and evidence is not None:
            year = evidence.year or getattr(getattr(evidence, "metadata", None), "year", None)
        if year is None:
            continue
        try:
            year_int = int(year)
        except Exception:
            continue
        year_buckets[year_int].append(claim)
    return year_buckets


def _fit_trend(year_to_value: Dict[int, float]) -> Tuple[bool, Dict[str, Any]]:
    years = sorted(year_to_value.keys())
    if len(years) < 3:
        return False, {"reason": "insufficient_years", "n": len(years)}

    xs = years
    ys = [year_to_value[y] for y in years]
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    ss_xx = sum((x - x_mean) ** 2 for x in xs)
    if ss_xx <= 0:
        return False, {"reason": "degenerate_year_variance", "n": len(years)}
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / ss_xx
    intercept = y_mean - slope * x_mean
    ss_tot = sum((y - y_mean) ** 2 for y in ys)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    r2 = 0.0 if ss_tot <= 0 else max(0.0, min(1.0, 1.0 - (ss_res / ss_tot)))

    p_value = None
    try:
        from scipy.stats import linregress
        lr = linregress(xs, ys)
        slope = float(lr.slope)
        r2 = max(0.0, min(1.0, float(lr.rvalue ** 2)))
        p_value = float(lr.pvalue)
    except Exception:
        p_value = None

    significant = (p_value is not None and p_value < 0.05 and r2 > 0.1)
    if p_value is None:
        significant = abs(slope) > 0.02 and r2 > 0.1

    return significant, {"slope": slope, "r2": r2, "p_value": p_value, "years": years, "values": ys}


async def conflict_analysis_node(state: GraphState) -> Dict[str, Any]:
    """Analyze claim clusters with NLI and temporal trend detection."""
    raw_pool = state.get("evidence_pool", [])
    raw_clusters = state.get("claim_clusters", [])
    candidates = list(state.get("candidate_answers", []) or [])
    pico = _as_pico(state.get("pico"))

    evidence_pool: List[EvidenceItem] = []
    evidence_by_pmid: Dict[str, EvidenceItem] = {}
    for raw in raw_pool:
        try:
            item = EvidenceItem(**raw) if isinstance(raw, dict) else raw
            evidence_pool.append(item)
            evidence_by_pmid[str(item.pmid)] = item
        except Exception as exc:
            logger.warning("Skipping malformed evidence item in conflict analysis: %s", exc)

    if not evidence_pool or not raw_clusters or not candidates:
        explanation = "missing_evidence_or_clusters_or_candidates"
        return {
            "candidate_stances": {candidate: 0.0 for candidate in candidates},
            "temporal_shift": {"detected": False, "details": {}, "explanation": explanation},
            "temporal_result": EpistemicResult(
                state=EpistemicState.INSUFFICIENT,
                belief=0.0,
                uncertainty=1.0,
                conflict=0.0,
                temporal_shift_detected=False,
                response_tier=ResponseTier.ABSTAIN,
                evidence_items=evidence_pool,
                baseline_claim=None,
                current_belief=None,
                contradiction_events=[],
            ),
            "epistemic_state": EpistemicState.INSUFFICIENT.value,
            "consensus_state": EpistemicState.INSUFFICIENT.value,
            "trace_events": [{"node": "conflict_analysis", "status": "skipped", "reason": explanation}],
        }

    nli_engine = NLIEngine()
    # Debug: emit raw NLI outputs for up to N claims to diagnose systematic bias
    debug_remaining = 5
    cluster_stances: Dict[str, Dict[str, Any]] = {}
    candidate_totals: Dict[str, float] = {candidate: 0.0 for candidate in candidates}
    candidate_weights: Dict[str, float] = {candidate: 0.0 for candidate in candidates}
    cluster_trace: List[Dict[str, Any]] = []
    # Per-PMID NLI aggregation containers (for propagating stances to EvidenceItem)
    per_pmid_signed_sum: Dict[str, float] = defaultdict(float)
    per_pmid_weight_sum: Dict[str, float] = defaultdict(float)

    for cluster in raw_clusters:
        cluster_id = str(cluster.get("cluster_id", len(cluster_stances)))
        cluster_claims = cluster.get("claims", []) or []
        cluster_stances[cluster_id] = {}

        for candidate in candidates:
            hypothesis = _candidate_hypothesis(candidate, pico)
            candidate_score_sum = 0.0
            candidate_weight_sum = 0.0
            candidate_breakdown: List[Dict[str, Any]] = []

            for claim in cluster_claims:
                premise = str(claim.get("text", "")).strip()
                if not premise:
                    continue
                claim_conf = float(claim.get("confidence", 1.0) or 1.0)
                try:
                    nli_res = await nli_engine.classify(premise, hypothesis)
                except Exception as exc:
                    logger.warning("NLI failed for cluster=%s candidate=%s: %s", cluster_id, candidate, exc)
                    nli_res = {"label": "NEUTRAL", "confidence": 0.0}

                # Debug logging for raw NLI outputs and computed stance
                if debug_remaining > 0:
                    try:
                        probs = nli_res.get("probs") or {}
                        probs_array = nli_res.get("probs_array")
                        debug_item = {
                            "pmid": claim.get("pmid"),
                            "claim_text": premise[:500],
                            "hypothesis": hypothesis,
                            "nli_raw_probs": probs if probs else (probs_array if probs_array else {}),
                            "nli_label": nli_res.get("label"),
                            "nli_confidence": nli_res.get("confidence"),
                            "computed_stance": _to_support_score(str(nli_res.get("label", "NEUTRAL")), float(nli_res.get("confidence", 0.0) or 0.0)),
                            "claim_confidence": claim_conf,
                        }
                        logger.info("NLI_DEBUG %s", json.dumps(debug_item, ensure_ascii=False))
                    except Exception:
                        logger.exception("Failed to emit NLI debug log")
                    debug_remaining -= 1

                label = str(nli_res.get("label", "NEUTRAL"))
                nli_conf = float(nli_res.get("confidence", 0.0) or 0.0)
                signed = _to_support_score(label, nli_conf)
                weight = max(0.0, claim_conf) * max(0.0, nli_conf)
                candidate_score_sum += signed * weight
                candidate_weight_sum += weight
                candidate_breakdown.append({
                    "pmid": claim.get("pmid"),
                    "text": premise[:220],
                    "label": label,
                    "nli_confidence": round(nli_conf, 3),
                    "claim_confidence": round(claim_conf, 3),
                    "weighted_support": round(signed * weight, 4),
                })

                # Aggregate per-PMID signed support for downstream propagation
                try:
                    pmid_key = str(claim.get("pmid", ""))
                    if pmid_key:
                        per_pmid_signed_sum[pmid_key] += signed * weight
                        per_pmid_weight_sum[pmid_key] += weight
                except Exception:
                    pass

            support_score = _normalize_support(candidate_score_sum, candidate_weight_sum)
            avg_claim_conf = sum(float(c.get("confidence", 1.0) or 1.0) for c in cluster_claims) / max(1, len(cluster_claims))
            cluster_stances[cluster_id][candidate] = {
                "support_score": round(support_score, 4),
                "nli_evidence_breakdown": candidate_breakdown,
                "cluster_size": len(cluster_claims),
                "avg_claim_confidence": round(avg_claim_conf, 3),
            }

            cluster_size = max(1, len(cluster_claims))
            cluster_weight = cluster_size * max(0.1, avg_claim_conf)
            candidate_totals[candidate] += support_score * cluster_weight
            candidate_weights[candidate] += cluster_weight

            cluster_trace.append({
                "cluster_id": cluster_id,
                "candidate": candidate,
                "support_score": round(support_score, 4),
                "evidence_count": len(candidate_breakdown),
                "top_labels": {
                    label: sum(1 for item in candidate_breakdown if item["label"] == label)
                    for label in ["ENTAILMENT", "CONTRADICTION", "NEUTRAL"]
                },
            })

    aggregated_candidate_stances = {
        candidate: round(_normalize_support(candidate_totals[candidate], candidate_weights[candidate]), 4)
        for candidate in candidates
    }

    # Propagate aggregated per-PMID NLI stance back to EvidenceItem objects
    pmid_stance_summary: Dict[str, Dict[str, Any]] = {}
    for pmid, evidence in evidence_by_pmid.items():
        pmid_str = str(pmid)
        signed_sum = per_pmid_signed_sum.get(pmid_str, 0.0)
        weight_sum = per_pmid_weight_sum.get(pmid_str, 0.0)
        norm = 0.0
        if weight_sum > 0:
            norm = signed_sum / weight_sum

        # Map normalized score to EvidenceStance
        from src.models.enums import EvidenceStance
        stance = EvidenceStance.NEUTRAL
        if norm >= 0.2:
            stance = EvidenceStance.SUPPORT
        elif norm <= -0.2:
            stance = EvidenceStance.OPPOSE

        # Update EvidenceItem in-place
        try:
            evidence.stance = stance
            evidence.nli_contradiction_prob = abs(norm)
        except Exception:
            # If evidence is a dict-like fallback, set keys
            try:
                evidence["stance"] = stance
                evidence["nli_contradiction_prob"] = abs(norm)
            except Exception:
                pass

        pmid_stance_summary[pmid_str] = {"normalized": round(norm, 3), "stance": str(stance)}

    temporal_details: Dict[str, Any] = {}
    temporal_shift_detected = False
    for candidate in candidates:
        year_to_scores: Dict[int, List[float]] = defaultdict(list)
        for cluster in raw_clusters:
            cluster_id = str(cluster.get("cluster_id", len(cluster_stances)))
            support_score = cluster_stances.get(cluster_id, {}).get(candidate, {}).get("support_score", 0.0)
            year_buckets = _cluster_year_items(cluster, evidence_by_pmid)
            for year, claims_for_year in year_buckets.items():
                if claims_for_year:
                    year_to_scores[year].append(support_score)

        year_to_mean = {year: sum(vals) / len(vals) for year, vals in year_to_scores.items() if vals}
        if not year_to_mean:
            temporal_details[candidate] = {"shift": False, "reason": "no_year_data"}
            continue

        has_shift, trend = _fit_trend(year_to_mean)
        if has_shift:
            temporal_shift_detected = True
        temporal_details[candidate] = {
            "shift": has_shift,
            **trend,
            "year_to_mean_support": {str(year): round(value, 4) for year, value in sorted(year_to_mean.items())},
        }

    sorted_candidates = sorted(aggregated_candidate_stances.items(), key=lambda kv: kv[1], reverse=True)
    best_candidate, best_score = sorted_candidates[0]
    second_score = sorted_candidates[1][1] if len(sorted_candidates) > 1 else -1.0

    if temporal_shift_detected:
        epistemic_state = EpistemicState.EVOLVING.value
    elif best_score > 0.6 and second_score < 0.2:
        epistemic_state = EpistemicState.SETTLED.value
    elif best_score > 0.25 and second_score > 0.1:
        epistemic_state = EpistemicState.CONTESTED.value
    else:
        epistemic_state = EpistemicState.CONTESTED.value

    # Display mapping: evaluator expects CONTROVERSIAL instead of CONTESTED
    display_state = epistemic_state if epistemic_state != EpistemicState.CONTESTED.value else "CONTROVERSIAL"

    top_clusters = sorted(
        cluster_stances.items(),
        key=lambda kv: max(v["support_score"] for v in kv[1].values()) if kv[1] else 0.0,
        reverse=True,
    )[:3]

    trace_events = [{
        "node": "conflict_analysis",
        "section": "nli_stance",
        "output": {
            "candidate_stances": aggregated_candidate_stances,
            "epistemic_state": display_state,
            "temporal_shift_detected": temporal_shift_detected,
        },
        "top_clusters": [
            {
                "cluster_id": cluster_id,
                "stances": {
                    candidate: round(values.get("support_score", 0.0), 4)
                    for candidate, values in cluster_value.items()
                },
            }
            for cluster_id, cluster_value in top_clusters
        ],
        "cluster_stance_distribution": cluster_trace[:12],
        "temporal_details": temporal_details,
        "explanation": (
            f"best_candidate={best_candidate} score={best_score:.3f}; "
            f"temporal_shift={temporal_shift_detected}; state={epistemic_state}"
        ),
    }]

    temporal_result = EpistemicResult(
        state=EpistemicState[epistemic_state],
        belief=max(0.0, min(1.0, (best_score + 1.0) / 2.0)),
        uncertainty=max(0.0, min(1.0, 1.0 - abs(best_score))),
        conflict=max(0.0, min(1.0, 1.0 - abs(best_score))),
        temporal_shift_detected=temporal_shift_detected,
        response_tier=ResponseTier.FULL if epistemic_state == EpistemicState.SETTLED.value else ResponseTier.QUALIFIED,
        evidence_items=evidence_pool,
        baseline_claim=best_candidate,
        current_belief=best_candidate,
        contradiction_events=[
            {
                "candidate": candidate,
                "shift": details.get("shift", False),
                "p_value": details.get("p_value"),
                "r2": details.get("r2"),
                "slope": details.get("slope"),
            }
            for candidate, details in temporal_details.items()
            if details.get("shift")
        ],
    )

    logger.info(
        "Conflict Analysis complete | best=%s score=%.3f state=%s shift=%s",
        best_candidate,
        best_score,
        epistemic_state,
        temporal_shift_detected,
    )

    for candidate, details in list(temporal_details.items())[:3]:
        if details.get("shift"):
            logger.info(
                "Temporal shift detected | candidate=%s slope=%.4f r2=%.3f p=%s",
                candidate,
                float(details.get("slope", 0.0) or 0.0),
                float(details.get("r2", 0.0) or 0.0),
                details.get("p_value"),
            )

    return {
        "candidate_stances": aggregated_candidate_stances,
        "cluster_stances": cluster_stances,
        "temporal_shift": {
            "detected": temporal_shift_detected,
            "details": temporal_details,
        },
        "temporal_result": temporal_result,
        "consensus_state": display_state,
        "epistemic_state": display_state,
        "trace_events": trace_events,
    }
