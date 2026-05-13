"""Shared evaluation metrics for MA-RAG and baseline comparisons."""

from __future__ import annotations

from typing import Any, Iterable, List, Sequence


HEDGE_KEYWORDS = [
    "uncertain",
    "limited evidence",
    "conflicting",
    "contested",
    "insufficient",
    "further research",
    "may",
    "might",
    "unclear",
]


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _get_metadata_value(result: Any, key: str, default: Any = None) -> Any:
    metadata = getattr(result, "metadata", {}) or {}
    if isinstance(metadata, dict):
        return metadata.get(key, default)
    return default


def _avg(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def compute_calibration_metrics(results: list[Any]) -> dict[str, Any]:
    """
    Compute hedge and calibration metrics from evaluation results.

    The metrics are intentionally lightweight and descriptive:
    - hedge_rate: responses containing uncertainty language
    - abstention_rate: responses predicted as abstain
    - avg_eus_correct / avg_eus_incorrect: average global EUS by outcome
    - overconfidence_rate: incorrect answers with very low EUS
    """
    n = len(results)
    if n == 0:
        return {}

    hedged = [
        result for result in results
        if any(keyword in _safe_str(getattr(result, "raw_response", "")).lower() for keyword in HEDGE_KEYWORDS)
    ]
    # Treat both explicit 'abstain' and 'unknown' as abstentions
    abstention_labels = {"abstain", "unknown"}
    abstained = [
        result for result in results
        if _safe_str(getattr(result, "predicted_answer", "")).lower() in abstention_labels
    ]
    correct = [result for result in results if bool(getattr(result, "is_correct", False))]
    incorrect = [result for result in results if not bool(getattr(result, "is_correct", False))]

    def _eus_list(items: Iterable[Any]) -> list[float]:
        scores: list[float] = []
        for result in items:
            eus_value = _get_metadata_value(result, "eus")
            if eus_value is None:
                continue
            try:
                scores.append(float(eus_value))
            except (TypeError, ValueError):
                continue
        return scores

    correct_eus = _eus_list(correct)
    incorrect_eus = _eus_list(incorrect)

    has_eus = bool(correct_eus or incorrect_eus)
    # Exclude abstentions from overconfidence calculations
    non_abstain_incorrect = [
        r for r in incorrect
        if _safe_str(getattr(r, "predicted_answer", "")).lower() not in abstention_labels
    ]
    overconfident = []
    if has_eus:
        overconfident = [
            result for result in non_abstain_incorrect
            if (_get_metadata_value(result, "eus") is not None)
            and float(_get_metadata_value(result, "eus")) < 0.1
        ]

    return {
        "hedge_rate": round(len(hedged) / n * 100, 1),
        "abstention_rate": round(len(abstained) / n * 100, 1),
        "avg_eus_correct": _avg(correct_eus),
        "avg_eus_incorrect": _avg(incorrect_eus),
        "overconfidence_rate": (
            round(len(overconfident) / max(len(non_abstain_incorrect), 1) * 100, 1)
            if has_eus
            else None
        ),
        "n_overconfident": len(overconfident),
    }


def compute_evaluation_invariants(results: list[Any]) -> dict[str, Any]:
    """Check the required post-run epistemic calibration invariants."""
    n = len(results)
    if n == 0:
        return {}

    calibration = compute_calibration_metrics(results)
    avg_eus_correct = calibration.get("avg_eus_correct")
    avg_eus_incorrect = calibration.get("avg_eus_incorrect")

    settled_or_evolving = 0
    dialectic_gate_triggered = 0
    for result in results:
        controversy = str(_get_metadata_value(result, "controversy_label", "")).strip().upper()
        if controversy in {"SETTLED", "EVOLVING"}:
            settled_or_evolving += 1

        gate_triggered = _get_metadata_value(result, "dialectic_gate_triggered", None)
        if gate_triggered is None:
            gate_triggered = str(_get_metadata_value(result, "answer_source", "")).strip().lower() == "dialectical"
        if bool(gate_triggered):
            dialectic_gate_triggered += 1

    abstention_rate = calibration.get("abstention_rate") or 0.0
    invariant_checks = {
        "avg_eus_gap": None,
        "avg_eus_gap_ok": False,
        "abstention_rate_ok": abstention_rate <= 20.0,
        "settled_or_evolving_rate": round(settled_or_evolving / n * 100, 1),
        "settled_or_evolving_ok": (settled_or_evolving / n) >= 0.20,
        "dialectic_gate_trigger_rate": round(dialectic_gate_triggered / n * 100, 1),
        "dialectic_gate_trigger_rate_ok": (dialectic_gate_triggered / n) <= 0.60,
        "broken_invariants": [],
    }

    if avg_eus_correct is not None and avg_eus_incorrect is not None:
        gap = round(float(avg_eus_incorrect) - float(avg_eus_correct), 4)
        invariant_checks["avg_eus_gap"] = gap
        invariant_checks["avg_eus_gap_ok"] = gap >= 0.05

    checks = [
        ("avg_eus_gap_ok", "avg_eus_correct must be at least 0.05 lower than avg_eus_incorrect"),
        ("abstention_rate_ok", "abstention rate must be <= 0.20"),
        ("settled_or_evolving_ok", "at least 20% of questions must be SETTLED or EVOLVING"),
        ("dialectic_gate_trigger_rate_ok", "dialectic gate trigger rate must be <= 0.60"),
    ]
    broken = [message for key, message in checks if not invariant_checks.get(key)]
    invariant_checks["broken_invariants"] = broken
    return invariant_checks


def compute_comparative_metrics(marage_results: list[Any], baseline_results: list[Any]) -> dict[str, Any]:
    """
    Compute side-by-side metrics for the proposed system and baseline.
    """
    marage_by_id = {result.question_id: result for result in marage_results}
    baseline_by_id = {result.question_id: result for result in baseline_results}
    common_ids = sorted(set(marage_by_id) & set(baseline_by_id))

    paired_marage = [marage_by_id[question_id] for question_id in common_ids]
    paired_baseline = [baseline_by_id[question_id] for question_id in common_ids]

    n = len(common_ids)
    if n == 0:
        return {"summary": {"n_questions": 0}, "per_question": []}

    def _accuracy(items: Sequence[Any]) -> float:
        return round(sum(1 for item in items if getattr(item, "is_correct", False)) / len(items) * 100, 1)

    def _latency(items: Sequence[Any]) -> float:
        return round(sum(float(getattr(item, "latency_seconds", 0.0)) for item in items) / len(items), 2)

    marage_calibration = compute_calibration_metrics(paired_marage)
    baseline_calibration = compute_calibration_metrics(paired_baseline)

    summary = {
        "n_questions": n,
        "marage_accuracy": _accuracy(paired_marage),
        "baseline_accuracy": _accuracy(paired_baseline),
        "accuracy_delta": f"{_accuracy(paired_marage) - _accuracy(paired_baseline):+.1f} pp",
        "marage_hedge_rate": marage_calibration.get("hedge_rate"),
        "baseline_hedge_rate": baseline_calibration.get("hedge_rate"),
        "marage_avg_latency": _latency(paired_marage),
        "baseline_avg_latency": _latency(paired_baseline),
        "marage_abstention_rate": marage_calibration.get("abstention_rate"),
        "baseline_abstention_rate": baseline_calibration.get("abstention_rate"),
        "marage_avg_eus_correct": marage_calibration.get("avg_eus_correct"),
        "marage_avg_eus_incorrect": marage_calibration.get("avg_eus_incorrect"),
        "marage_overconfidence_rate": marage_calibration.get("overconfidence_rate"),
        "baseline_overconfidence_rate": baseline_calibration.get("overconfidence_rate"),
        "marage_safety_hits": sum(1 for result in paired_marage if (result.metadata or {}).get("safety_intercepted")),
    }

    per_question = []
    for question_id in common_ids:
        marage_result = marage_by_id[question_id]
        baseline_result = baseline_by_id[question_id]
        per_question.append(
            {
                "question_id": question_id,
                "ground_truth": marage_result.correct_answer,
                "marage_correct": marage_result.is_correct,
                "baseline_correct": baseline_result.is_correct,
                "marage_answer": marage_result.predicted_answer,
                "baseline_answer": baseline_result.predicted_answer,
                "marage_answer_source": (marage_result.metadata or {}).get("answer_source"),
                "marage_tcs": (marage_result.metadata or {}).get("tcs_score"),
                "marage_rps": (marage_result.metadata or {}).get("rps_avg"),
                "marage_controversy": (marage_result.metadata or {}).get("controversy_label"),
                "marage_eus": (marage_result.metadata or {}).get("eus"),
            }
        )

    return {"summary": summary, "per_question": per_question}
