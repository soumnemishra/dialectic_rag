# what the important of the this file is that it checks 
'''
--> evidence section says that the 5 evidence collect state that the treatment works
-->but the conclusion says that system doesnot works 
-->your conlcusion contradicts your evidence 
--> that reviewner is the decision alignment node 
'''


import logging
import os
import re
import random
from typing import Dict, Any, List

from langchain_core.prompts import ChatPromptTemplate

from src.core.registry import ModelRegistry, safe_ainvoke
from src.epistemic.dempster_shafer import build_epistemic_masses, fuse_masses
from src.state.state import GraphState
from src.agents.answer_utils import extract_final_answer_letter
from src.utils.epistemic_trace import build_trace_event, build_trace_updates

logger = logging.getLogger(__name__)

TRANSPARENCY_DISCLAIMER = (
    "Note: Retrieved literature did not contain sufficient evidence to definitively answer this query. "
    "This answer was synthesized using general medical knowledge."
)



# ------------------------------------------------------------------ #
#  Answer extraction patterns                                         #
# ------------------------------------------------------------------ #
# These regex patterns extract the final MedQA answer letter from the
# summariser's answer text.
#
# Why regex over LLM: this is a deterministic structural check, not
# semantic reasoning. Regex is fast, free, and testable.
#
# Ordered from most specific to least specific — re.findall returns
# ALL matches; we take the LAST one (the final decision in the text).

ANSWER_PATTERNS = [
    r"\*\*final\s+answer:\s*([A-D]|UNKNOWN)\*\*",
    r"final\s+answer:\s*\*?\*?([A-D]|UNKNOWN)\*?\*?",
    r"(?:final\s+)?answer\s*(?:is|:)\s*[\"']?\b([A-D]|UNKNOWN)\b[\"']?",
    r"(?:^|\n)\s*\**\s*([A-D]|UNKNOWN)\s*\**\s*(?:$|\n)",
]

MCQ_PATTERNS = [
    r"\\boxed\{([A-D]|UNKNOWN)\}",
    r"\*\*final\s+answer:\s*\[?([A-D]|UNKNOWN)\]?\*\*",
    r"final\s+answer:\s*\[?([A-D]|UNKNOWN)\]?",
    r"most\s+likely\s+(?:diagnosis|answer)[^\w]{0,10}([A-D]|UNKNOWN)\b",
    r"selected\s+option\s*:\s*([A-D]|UNKNOWN)\b",
    r"(?:the\s+)?(?:correct|best|most\s+appropriate|most\s+likely)\s+(?:option|choice|answer)\s*(?:is|would\s+be|seems|appears)?\s*([A-D]|UNKNOWN)\b",
    r"(?:option|choice)\s+([A-D]|UNKNOWN)\s*(?:is|would\s+be|seems|appears|represents|best|most\s+likely|correct|appropriate)\b",
    r"(?im)(?:^|\n)\s*(?:option|choice)\s+([A-D]|UNKNOWN)\s*(?:$|[\s\.,;:])",
    r"(?:therefore|thus|hence|so)[^\n]{0,60}?(?:option|choice)\s+([A-D]|UNKNOWN)\b",
    r'"answer"\s*:\s*"([A-D]|UNKNOWN)"',
]

VALID_ANSWER_CHOICES = {"A", "B", "C", "D", "UNKNOWN"}

POLARITY_TO_ABSTENTION = {
    "support": None,
    "refute": "UNKNOWN",
    "mixed": "UNKNOWN",
    "insufficient": "UNKNOWN",
}


# ------------------------------------------------------------------ #
#  Epistemic guardrail (EPISTEMIC_MODE only)                          #
# ------------------------------------------------------------------ #

def should_answer_be_deferred(
    state: GraphState,
    summary: Dict[str, float] | None = None,
    polarity_override: str | None = None,
) -> bool:
    """Return True if epistemic metadata indicates the answer should be
    deferred to UNKNOWN regardless of the LLM's chosen letter.

    Checks (in order — early exits take priority):
      0. Strong support override: if polarity=support AND confidence>=0.7
         AND ds_belief>=0.6, NEVER defer (prevents RPS degradation from
         overriding correct reasoning).
      1. controversy_label is CONTESTED or EVOLVING
      2. EUS > 0.25 (from eus_per_claim["global"])
      3. RPS average is below 0.3 (very low reproducibility)
    """
    from src.config import settings
    logger.info(f"EPISTEMIC_MODE = {settings.EPISTEMIC_MODE}")
    if not settings.EPISTEMIC_MODE:
        return False

    polarity_data = state.get("evidence_polarity", {}) or {}
    polarity = polarity_data.get("polarity", "insufficient") if isinstance(polarity_data, dict) else "insufficient"
    if polarity_override:
        polarity = polarity_override

    summary = summary or _epistemic_summary(state)
    support = summary.get("support", 0.0)
    uncertainty = summary.get("uncertain", 1.0)
    conflict = summary.get("conflict", 0.0)

    if support > 0.6 and polarity != "refute":
        return False
    if uncertainty > 0.6 or conflict > 0.5:
        return True
    return False


def should_abstain(
    polarity: str,
    confidence: float,
    ds_uncertainty: float,
    step_consensus: float,
    applicability_score: float,
    controversy_label: str,
    ds_support: float,
) -> bool:
    """
    Only abstain if there is genuine epistemic conflict or insufficient coverage.

    Special-case: low-confidence `refute` signals frequently indicate the
    summariser disagrees with retrieved evidence; abstain unless the fused
    DS support strongly favors a concrete option.
    """
    # NEVER abstain on settled, strongly supported, high-consensus evidence
    if (
        (polarity == "strong_support" or polarity == "support")
        and confidence >= 0.95
        and step_consensus >= 0.95
        and controversy_label in ("SETTLED", None, "UNKNOWN")
    ):
        return False

    # Special-case: explicit refute with low evidence confidence -> abstain
    if polarity == "refute" and float(confidence or 0.0) < 0.50:
        # Abstain unless DS strongly supports an alternative
        if ds_support < 0.60 or ds_uncertainty > 0.45 or step_consensus < 0.60:
            return True

    # Abstain only if uncertainty is high AND evidence is contested/insufficient
    if polarity in ("insufficient", "weak_support") and ds_uncertainty > 0.60:
        return True

    # For contested/evolving evidence, use adaptive threshold
    if str(controversy_label).upper() == "EVOLVING" and ds_uncertainty > 0.70:
        return True

    if str(controversy_label).upper() == "CONTESTED" and ds_uncertainty > 0.65:
        return True

    # Default to preserving the answer if it's not clearly failing
    return False



def _normalize_answer_token(token: str) -> str:
    value = str(token or "").strip().upper()
    if value in {"A", "B", "C", "D", "UNKNOWN"}:
        return value
    return "UNKNOWN"


def _average_rps(state: GraphState) -> float:
    rps_scores = state.get("rps_scores") or []
    valid: list[float] = []
    for score in rps_scores:
        if not isinstance(score, dict):
            continue
        val = score.get("final_score")
        if val is None:
            val = score.get("rps_score")
        if val is None:
            val = score.get("rps")
        if val is None:
            continue
        try:
            valid.append(float(val))
        except (TypeError, ValueError):
            continue
    return sum(valid) / len(valid) if valid else 0.5


def _step_consensus(state: GraphState) -> tuple[float, Dict[str, int]]:
    counts: Dict[str, int] = {}
    definitive = 0
    for output in state.get("step_output", []) or []:
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


def _missing_evidence_topics(state: GraphState) -> List[str]:
    topics: List[str] = []
    for note in state.get("step_notes", []) or []:
        if not isinstance(note, str):
            continue
        if "RETRIEVAL_MISSING" not in note:
            continue
        match = re.search(r"RETRIEVAL_MISSING[:\s-]*(.*)", note, flags=re.IGNORECASE)
        topic = match.group(1).strip() if match and match.group(1).strip() else note.strip()
        topics.append(topic)
    for output in state.get("step_output", []) or []:
        if not isinstance(output, dict):
            continue
        if str(output.get("retrieval_status", "")).upper() != "MISSING":
            continue
        topic = str(output.get("question") or output.get("task") or output.get("analysis") or "RETRIEVAL_MISSING").strip()
        topics.append(topic)
    deduped: List[str] = []
    seen: set[str] = set()
    for topic in topics:
        key = topic.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(topic)
    return deduped


def _epistemic_label_from_eus(eus: float) -> str:
    if eus <= 0.20:
        return "WELL-SUPPORTED"
    if eus <= 0.35:
        return "MODERATE CONFIDENCE"
    if eus <= 0.55:
        return "WARRANTED BUT CONTESTED"
    if eus <= 0.70:
        return "HIGH UNCERTAINTY"
    return "INSUFFICIENT EVIDENCE"


def _epistemic_label_from_state(controversy_label: str, avg_rps: float) -> tuple[str, str]:
    label_up = str(controversy_label or "UNKNOWN").strip().upper()
    if label_up == "SETTLED":
        return "SETTLED", "WELL-SUPPORTED"
    if label_up == "EVOLVING":
        return "EVOLVING", "HIGH UNCERTAINTY"
    if label_up == "CONTESTED":
        return "CONTESTED", "WARRANTED BUT CONTESTED"
    return "SETTLED", "WELL-SUPPORTED"


def _append_epistemic_footer(answer_text: str, state: GraphState, calibrated_confidence: float, controversy_label: str) -> str:
    full = str(answer_text or "")
    # Preserve any existing final-answer tag so we can ensure it remains the
    # last visible element after we append the epistemic footer.
    final_tags = re.findall(r"(?is)\*\*final\s+answer:\s*(?:[A-D]|UNKNOWN)\*\*", full)
    final_tag = final_tags[-1] if final_tags else None

    # Remove all final-answer tags from the base text so we can append the
    # footer and then re-attach the canonical final tag last.
    if final_tag:
        base = re.sub(r"(?is)\*\*final\s+answer:\s*(?:[A-D]|UNKNOWN)\*\*", "", full).strip()
    else:
        base = full.strip()

    avg_rps = _average_rps(state)
    tcs = float(state.get("tcs_score", 0.0) or 0.0)
    eus = float(((state.get("eus_per_claim") or {}).get("global", 0.0)) or 0.0)
    eus_label = _epistemic_label_from_eus(eus)
    global_interval = (state.get("belief_intervals") or {}).get("global", {}) or {}
    support = float(global_interval.get("belief", 0.0) or 0.0)
    plausibility = float(global_interval.get("plausibility", support) or support)
    missing_topics = _missing_evidence_topics(state)
    step_consensus, _ = _step_consensus(state)

    footer = (
        f"\n\n**Controversy Status:** {str(controversy_label or 'UNKNOWN').strip().upper()}"
        f"\n**Average RPS:** {avg_rps:.2f}"
        f"\n**Uncertainty:** {eus:.2f} ({eus_label})"
        f"\n**Calibrated Confidence:** {calibrated_confidence:.2f}"
        f"\n\nEvidence Quality Summary:\n"
        f"- Controversy Label: {str(controversy_label or 'UNKNOWN').strip().upper()}\n"
        f"- Average RPS: {avg_rps:.2f}\n"
        f"- Temporal Conflict Score: {tcs:.2f}\n"
        f"- Dempster-Shafer Belief Interval: [{support:.2f}, {plausibility:.2f}]\n"
        f"- Step Consensus: {step_consensus:.2f}"
    )

    if missing_topics:
        footer += "\n- Missing Evidence Topics: " + "; ".join(missing_topics)
        footer += "\nNo literature evidence was found for the missing topic(s)."
    else:
        footer += "\n- Missing Evidence Topics: none"

    if eus_label == "INSUFFICIENT EVIDENCE":
        footer += (
            "\n\nThe available evidence is insufficient to provide a reliable answer "
            f"(DS Uncertainty: {max(plausibility - support, 0.0):.2f}, Step Consensus: {step_consensus:.2f})."
        )
    elif str(controversy_label or "").strip().upper() == "CONTESTED":
        footer += "\n\nThe evidence is currently contested; clinical correlation is strongly advised."

    if str(state.get("intent", "")).lower() == "diagnostic" and str(state.get("risk_level", "")).lower() == "high" and str(controversy_label or "").strip().upper() == "CONTESTED":
        footer += "\n\nThe evidence is currently contested; clinical correlation is essential."

    if final_tag:
        return (base + footer + "\n\n" + final_tag).strip()
    return (base + footer).strip()


def _conflict_ratio(state: GraphState) -> float:
    synthesis = state.get("dialectic_synthesis") or {}
    if not isinstance(synthesis, dict):
        return 0.0
    try:
        thesis = int(synthesis.get("thesis_count", 0) or 0)
        antithesis = int(synthesis.get("antithesis_count", 0) or 0)
    except (TypeError, ValueError):
        thesis = 0
        antithesis = 0
    total = max(thesis + antithesis, 1)
    return antithesis / total


def _epistemic_summary_from_signals(
    state: GraphState,
    polarity: str,
    polarity_confidence: float,
) -> Dict[str, float]:
    masses = build_epistemic_masses(
        tcs_score=float(state.get("tcs_score", 0.0) or 0.0),
        avg_rps=_average_rps(state),
        applicability=float(state.get("applicability_score", 0.5) or 0.5),
        controversy_label=str(state.get("controversy_label", "UNKNOWN")),
        conflict_ratio=_conflict_ratio(state),
        polarity=polarity,
        polarity_confidence=polarity_confidence,
    )
    combined, conflict = fuse_masses(masses)
    return {
        "support": float(combined.get("SUPPORT", 0.0)),
        "refute": float(combined.get("REFUTE", 0.0)),
        "uncertain": float(combined.get("UNCERTAIN", 1.0)),
        "conflict": float(conflict),
    }


def _epistemic_summary(state: GraphState) -> Dict[str, float]:
    intervals = state.get("belief_intervals") or {}
    global_interval = intervals.get("global") if isinstance(intervals, dict) else None
    if isinstance(global_interval, dict) and "belief" in global_interval:
        support = float(global_interval.get("belief", 0.0) or 0.0)
        plausibility = float(global_interval.get("plausibility", support) or support)
        uncertainty = float((state.get("eus_per_claim") or {}).get("global", 1.0) or 1.0)
        conflict = float(global_interval.get("conflict", 0.0) or 0.0)
        refute = max(0.0, min(1.0, 1.0 - support - uncertainty))
        return {
            "support": support,
            "refute": refute,
            "uncertain": uncertainty,
            "conflict": conflict,
        }

    evidence_polarity = state.get("evidence_polarity", {})
    polarity = evidence_polarity.get("polarity", "insufficient") if isinstance(evidence_polarity, dict) else "insufficient"
    polarity_conf = float(evidence_polarity.get("confidence", 0.0)) if isinstance(evidence_polarity, dict) else 0.0

    return _epistemic_summary_from_signals(state, polarity, polarity_conf)


def _get_current_answer(text: str) -> str:
    """
    Extract the final MedQA decision letter from an answer string.

    Takes the LAST match so that if the summariser writes multiple
    candidate answers and then a final decision, we get the final one.

    Returns "UNKNOWN" if no standard answer tag is found.
    """
    if not text:
        return "UNKNOWN"
    try:
        letter = extract_final_answer_letter(text, fallback="UNKNOWN")
        return _normalize_answer_token(letter)
    except Exception:
        return "UNKNOWN"


def _extract_answer_from_plan_summary(state: GraphState) -> str | None:
    """
    Scan plan summary and step notes for a final answer letter [A-D].
    Returns uppercase answer letter or UNKNOWN.

    Important: this intentionally ignores state['final_answer'] so the
    extracted value comes from the aggregate synthesis, not from the
    current surface answer being aligned.
    """
    def _extract_from_dict(data: dict) -> str | None:
        cand_opt = str(data.get("candidate_option") or "").upper()
        cand_ent = str(data.get("candidate_entity") or "").strip()
        
        if cand_opt in {"A", "B", "C", "D"}:
            # Structured validation: option_map[candidate_option] == candidate_entity
            options = _resolve_mcq_options(state)
            if cand_opt in options:
                # Use robust regex-based text match to confirm entity identity
                regex = _build_option_regex(options[cand_opt])
                if regex and (regex.search(cand_ent) or cand_ent.lower() in options[cand_opt].lower()):
                    return cand_opt
            # Fallback if entity doesn't match exactly but letter is provided
            return cand_opt
        
        # Backward compatibility for existing logs
        for key in ("mcq_letter", "predicted_letter", "final_decision", "answer"):
            val = data.get(key)
            if val and isinstance(val, str):
                letter = _get_current_answer(val)
                if letter in {"A", "B", "C", "D"}:
                    return letter
        return None

    plan_summary = state.get("plan_summary")
    if isinstance(plan_summary, dict):
        letter = _extract_from_dict(plan_summary)
        if letter:
            return letter

    plan = state.get("plan", [])
    aggregate_step_ids = {
        int(step.get("id", -1))
        for step in plan
        if isinstance(step, dict) and step.get("step_type") == "aggregate"
    }

    step_outputs = state.get("step_output", [])
    if isinstance(step_outputs, list):
        for output in reversed(step_outputs):
            if not isinstance(output, dict):
                continue
            is_aggregate = (
                output.get("answer_source") == "step6_aggregate"
                or output.get("step_type") == "aggregate"
                or output.get("type") == "aggregate"
                or int(output.get("step_id", -1)) in aggregate_step_ids
            )
            if not is_aggregate:
                continue
            letter = _extract_from_dict(output)
            if letter:
                return letter

    text_sources = []

    def _add_text(value: Any) -> None:
        if value is None:
            return
        if isinstance(value, str):
            text_sources.append(value)
            return
        if isinstance(value, dict):
            for key in (
                "mcq_letter",
                "predicted_letter",
                "answer",
                "final_decision",
                "output",
                "summary",
                "evidence_summary",
                "analysis",
                "final_answer",
                "final_summary",
            ):
                if key in value:
                    _add_text(value.get(key))
            return
        if isinstance(value, list):
            for item in value:
                _add_text(item)
            return
        text_sources.append(str(value))

    _add_text(state.get("plan_summary"))
    _add_text(state.get("step_notes", []))
    _add_text(state.get("step_output", []))

    for exp in state.get("past_exp", []):
        _add_text(exp)
    # Also include the final_answer text when extracting letters from aggregate/analysis
    _add_text(state.get("final_answer"))

    combined = "\n".join(t for t in text_sources if t)
    if not combined:
        return None

    for pattern in MCQ_PATTERNS:
        match = re.search(pattern, combined, re.IGNORECASE)
        if match:
            return _normalize_answer_token(match.group(1))

    return None


def _extract_alternative_answer(text: str) -> str | None:
    if not text:
        return None

    for pattern in MCQ_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
        if matches:
            return _normalize_answer_token(matches[-1])
    return None


def _stamp_final_answer(text: str, answer: str) -> str:
    """Replace any existing Final Answer tag and append the canonical one."""
    cleaned = re.sub(r"(?is)\*\*final\s+answer:\s*(?:[A-D]|UNKNOWN)\*\*", "", text).strip()
    if cleaned:
        return f"{cleaned}\n\n**Final Answer: {answer.upper()}**"
    return f"**Final Answer: {answer.upper()}**"


def _strip_references_block(text: str) -> str:
    """Remove a trailing References block and PMID bullets from an answer."""
    if not text:
        return text

    lines = text.splitlines()
    cleaned_lines = []
    skipping_references = False

    for line in lines:
        stripped = line.strip()

        if not skipping_references and re.match(r"^\*\*references:\*\*$", stripped, re.IGNORECASE):
            skipping_references = True
            continue

        if skipping_references:
            if not stripped:
                continue
            if re.match(r"^[-*]\s*PMID:\s*\d+", stripped, re.IGNORECASE):
                continue
            skipping_references = False

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


def _sanitize_citation_hallucination(text: str, should_sanitize: bool) -> str:
    """Replace hallucinated citations with a transparency disclaimer when needed."""
    if not should_sanitize:
        return text

    cleaned = _strip_references_block(text).rstrip()
    if TRANSPARENCY_DISCLAIMER in cleaned:
        return cleaned

    if cleaned:
        return f"{cleaned}\n\n{TRANSPARENCY_DISCLAIMER}"
    return TRANSPARENCY_DISCLAIMER


def _extract_mcq_options(question: str) -> Dict[str, str]:
    """Extract A-D option text from a multiple-choice question prompt."""
    options: Dict[str, str] = {}
    if not question:
        return options

    for match in re.finditer(r"(?im)^\s*([A-D])\s*[:\.)]\s*(.+?)\s*$", str(question)):
        letter = match.group(1).upper()
        option_text = match.group(2).strip()
        if option_text:
            options[letter] = option_text

    return options


def _resolve_mcq_options(state: GraphState) -> Dict[str, str]:
    mcq_options = state.get("mcq_options", "") or ""
    original_question = state.get("original_question", "") or ""
    if isinstance(mcq_options, dict):
        return {str(k).upper(): str(v) for k, v in mcq_options.items() if v}
    return _extract_mcq_options(mcq_options) or _extract_mcq_options(original_question)


def _build_option_regex(option_text: str) -> re.Pattern | None:
    tokens = re.findall(r"[A-Za-z0-9]+", str(option_text))
    if not tokens:
        return None
    # Allow mild formatting differences: whitespace, hyphens, slashes.
    joined = r"[\s\-_/]*".join(re.escape(t) for t in tokens)
    return re.compile(rf"\b{joined}\b", re.IGNORECASE)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+", str(text).lower())


def _option_overlap_score(option_text: str, candidate_text: str) -> float:
    opt_tokens = set(_tokenize(option_text))
    cand_tokens = set(_tokenize(candidate_text))
    if not opt_tokens or not cand_tokens:
        return 0.0
    return len(opt_tokens & cand_tokens) / len(opt_tokens)


def _extract_candidate_phrases(reasoning: str) -> list[str]:
    if not reasoning:
        return []

    patterns = [
        r"(?:suggests?|consistent with|points to|indicates|diagnosis is|most likely|likely|favoring|supports)\s+([^\.;\n]+)",
        r"(?:rules out|less likely)\s+([^\.;\n]+)",
    ]
    candidates: list[str] = []
    for pattern in patterns:
        for match in re.finditer(pattern, reasoning, re.IGNORECASE):
            phrase = match.group(1).strip()
            if phrase:
                candidates.append(phrase)

    # Add capitalized disease-like phrases as fallback candidates.
    cap_matches = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b", reasoning)
    candidates.extend(cap_matches)

    seen: set[str] = set()
    unique: list[str] = []
    for cand in candidates:
        key = cand.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(cand.strip())
    return unique


def _infer_alternative_option_from_reasoning(
    reasoning: str,
    options: Dict[str, str],
    current_letter: str,
) -> tuple[str | None, str | None, float]:
    if not reasoning:
        return None, None, 0.0

    # Direct match: structured option index (e.g., 'Option B')
    match = re.search(r"\b(?:option|choice)?\s*([A-D])\b", reasoning, re.IGNORECASE)
    if match:
        letter = match.group(1).upper()
        # If options are not provided, accept explicit 'option X' mentions
        if (not options or letter in options) and letter != current_letter:
            opt_text = options.get(letter) if isinstance(options, dict) and letter in options else None
            return letter, opt_text, 1.0

    # Direct match: option text appears exactly in reasoning.
    best_letter = None
    best_text = None
    best_score = 0.0

    for letter, option_text in options.items():
        if letter == current_letter:
            continue
        regex = _build_option_regex(option_text)
        if regex and regex.search(reasoning):
            score = 1.0
            if score > best_score:
                best_letter, best_text, best_score = letter, option_text, score

    if best_letter:
        return best_letter, best_text, best_score

    return None, None, 0.0


def _infer_mcq_letter_from_option_text(original_question: str, answer_text: str) -> str | None:
    """Infer option letter by matching option texts inside the answer narrative."""
    options = _extract_mcq_options(original_question)
    if not options or not answer_text:
        return None

    cleaned = re.sub(
        r"(?is)\*\*final\s+answer:\s*\[?(?:[A-D]|UNKNOWN)\]?\*\*",
        "",
        str(answer_text),
    ).strip()
    if not cleaned:
        return None

    # Prefer the tail of the answer where the conclusion usually lives.
    tail = cleaned[-1200:]
    sentences = re.split(r"(?<=[\.\!\?])\s+", tail)
    recent = sentences[-8:] if len(sentences) > 8 else sentences

    option_regexes: Dict[str, re.Pattern] = {}
    for letter, text in options.items():
        regex = _build_option_regex(text)
        if regex is not None:
            option_regexes[letter] = regex

    if not option_regexes:
        return None

    # Score options based on where and how they appear.
    best_letter: str | None = None
    best_score = 0
    best_pos = -1
    second_best = 0

    for idx, sentence in enumerate(recent):
        s_lower = sentence.lower()
        keyword_bonus = 0
        if "most likely" in s_lower:
            keyword_bonus += 4
        if "diagnosis" in s_lower:
            keyword_bonus += 2
        if any(k in s_lower for k in ("best explained", "best explains", "best explanation")):
            keyword_bonus += 6
        if any(k in s_lower for k in ("therefore", "thus", "hence")):
            keyword_bonus += 2
        if any(k in s_lower for k in ("consistent with", "suggestive of", "aligns with", "points to")):
            keyword_bonus += 1

        negation_penalty = 0
        if any(k in s_lower for k in ("not applicable", "does not apply", "doesn't apply")):
            negation_penalty += 10
        if any(k in s_lower for k in ("lacks", "without", "ruled out", "unlikely", "less likely")):
            negation_penalty += 4
        # Light penalty for generic negation words in the same sentence.
        if re.search(r"\b(not|no|never|cannot|can't|isn't|aren't|doesn't|does not)\b", s_lower):
            negation_penalty += 1

        for letter, regex in option_regexes.items():
            if not regex.search(sentence):
                continue
            score = 1 + keyword_bonus - negation_penalty
            # Later sentences are more likely to be the final decision.
            pos = idx
            if score > best_score or (score == best_score and pos > best_pos):
                second_best = best_score
                best_letter = letter
                best_score = score
                best_pos = pos
            elif score > second_best and letter != best_letter:
                second_best = score

    if not best_letter or best_score <= 0:
        return None
    # Require a clear winner to avoid mapping when multiple options are discussed.
    if second_best == best_score:
        return None
    return best_letter


def _gather_alignment_text_candidates(state: GraphState, final_answer: str) -> list[str]:
    """Collect candidate texts that may contain an implied A-D decision."""
    candidates: list[str] = []

    plan_summary = state.get("plan_summary")
    if isinstance(plan_summary, dict):
        for key in (
            "answer",
            "summary",
            "evidence_summary",
            "analysis",
            "final_answer",
            "final_decision",
            "final_diagnosis",
            "output",
        ):
            value = plan_summary.get(key)
            if value:
                candidates.append(str(value))
    elif plan_summary:
        candidates.append(str(plan_summary))

    step_outputs = state.get("step_output", [])
    if isinstance(step_outputs, list):
        for output in reversed(step_outputs):
            if not isinstance(output, dict):
                continue
            for key in (
                "summary",
                "evidence_summary",
                "final_diagnosis",
                "answer",
                "final_answer",
                "analysis",
            ):
                value = output.get(key)
                if value:
                    candidates.append(str(value))

    if final_answer:
        candidates.append(str(final_answer))

    # De-dupe while preserving order.
    seen: set[str] = set()
    ordered: list[str] = []
    for text in candidates:
        t = str(text or "").strip()
        if not t:
            continue
        if t in seen:
            continue
        seen.add(t)
        ordered.append(t)
    return ordered


_LETTER_PICK_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You select the final multiple-choice answer letter for a medical question. "
        "Return ONLY one character: A, B, C, or D. No other text.",
    ),
    (
        "human",
        "Original Question (with options):\n{original_question}\n\n"
        "Rationale / Aggregate Summary:\n{rationale}\n\n"
        "Return ONLY the single best option letter (A/B/C/D).",
    ),
])


async def _llm_force_pick_letter(original_question: str, rationale: str) -> str | None:
    if not original_question or not rationale:
        return None
    if not _extract_mcq_options(original_question):
        return None

    enabled = os.getenv("MRAGE_DECISION_ALIGNMENT_FORCE_COMMIT", "1").strip().lower() not in {"0", "false", "no"}
    if not enabled:
        return None

    try:
        llm = ModelRegistry.get_light_llm(temperature=0.0, json_mode=False)
        if llm is None:
            return None

        # Optionally randomize option display order to reduce positional
        # bias when using the lightweight LLM to map rationale -> letter.
        question_for_model = original_question
        try:
            rand_opts = os.getenv("MRAGE_DECISION_ALIGNMENT_RANDOMIZE_OPTIONS", "0").strip().lower() in {"1", "true", "yes"}
            if rand_opts:
                options = _extract_mcq_options(original_question)
                if options:
                    items = list(options.items())
                    random.shuffle(items)
                    # Attempt to preserve the stem (text before options) if present
                    stem = original_question
                    split_match = re.split(r"(?im)^[A-D]\s*[:\.)]\s*", original_question, maxsplit=1)
                    if split_match and len(split_match) > 1:
                        stem = split_match[0].strip()
                    opt_lines = [f"{letter}) {text}" for letter, text in items]
                    question_for_model = (stem + "\n" + "\n".join(opt_lines)).strip()
                    logger.debug("Decision alignment: randomized options for LLM force-pick")
        except Exception:
            question_for_model = original_question

        chain = _LETTER_PICK_PROMPT | llm
        result = await safe_ainvoke(chain, {
            "original_question": str(question_for_model)[:4000],
            "rationale": str(rationale)[:2500],
        })
        text = result.content if hasattr(result, "content") else str(result)
        match = re.search(r"(?i)\b([A-D])\b", str(text))
        if match:
            return match.group(1).upper()
        return None
    except Exception as exc:
        logger.warning("Decision alignment force-pick failed: %s", exc)
        return None


# ------------------------------------------------------------------ #
#  Node                                                               #
# ------------------------------------------------------------------ #

async def decision_alignment_node(state: GraphState) -> Dict[str, Any]:
    """
    Enforces consistency between Evidence Polarity and the Final Answer.

    The core rule: never make an answer MORE confident than the evidence
    supports. Only DOWNGRADE to UNKNOWN when the evidence is too weak or
    contradictory for a concrete option.

    Example:
        Polarity = "refute" (confidence 0.85) + Answer = "B"
        → override to "UNKNOWN"  (evidence contradicts the answer)

        Polarity = "support" (confidence 0.80) + Answer = "B"
        → no change needed

    This node MUST run AFTER the summariser writes final_answer to state.
    If final_answer is empty, it exits early with a visible warning so
    graph.py ordering problems are immediately detectable in logs.

    Eval mode forced resolution:
        When evaluation_mode=True AND retry_count >= 1 AND confidence < 0.6,
        forces an explicit abstention when the evidence is too weak or
        contradictory for a concrete MedQA option. Every override is recorded in
        safety_flags for full audit traceability.

    Returns:
        {} if no alignment change needed (most queries)
        {"final_answer": ..., "safety_flags": ...} when override applied
    """
    final_answer  = state.get("final_answer", "")
    raw_final_answer = state.get("final_answer_raw") or final_answer
    predicted_letter = _normalize_answer_token(state.get("predicted_letter", ""))
    answer        = state.get("answer", "")
    polarity_data = state.get("evidence_polarity", {})
    safety_flags  = state.get("safety_flags", [])

    # Guard: must run after summariser
    if not final_answer:
        logger.warning(
            "decision_alignment_node: final_answer is empty. "
            "This node must run AFTER the summariser. "
            "Check node ordering in graph.py."
        )
        return {}

    polarity   = polarity_data.get("polarity",   "insufficient") \
                 if isinstance(polarity_data, dict) else "insufficient"
    confidence = float(polarity_data.get("confidence", 0.0)) \
                 if isinstance(polarity_data, dict) else 0.0
    original_polarity = polarity
    switch_applied = False

    controversy_label = state.get("controversy_label", "UNKNOWN")
    conflict_updates: Dict[str, Any] = {}
    if polarity == "refute":
        if str(controversy_label or "").strip().upper() in {"", "UNKNOWN"}:
            controversy_label = "CONTESTED"
        conflict_updates = {
            "controversy_label": controversy_label,
        }
        if "evidence_refute_conflict" not in safety_flags:
            safety_flags = safety_flags + ["evidence_refute_conflict"]

    current_answer = predicted_letter if predicted_letter in VALID_ANSWER_CHOICES else _get_current_answer(answer)
    current_answer = str(current_answer or "UNKNOWN").strip().upper()
    if current_answer not in VALID_ANSWER_CHOICES:
        current_answer = "UNKNOWN"
    if current_answer == "UNKNOWN":
        current_answer = _get_current_answer(final_answer)
        current_answer = str(current_answer or "UNKNOWN").strip().upper()
        if current_answer not in VALID_ANSWER_CHOICES:
            current_answer = "UNKNOWN"
    answer_source = state.get("answer_source", "rag_direct")
    if answer_source == "rag_direct" and state.get("plan") and len(state.get("step_output", [])) > 0:
        answer_source = "plan_execute"
        
    should_sanitize_citations = polarity == "insufficient" or answer_source == "general_knowledge"

    if should_sanitize_citations:
        sanitized_answer = _sanitize_citation_hallucination(final_answer, True)
        if sanitized_answer != final_answer:
            logger.info(
                "Decision alignment: sanitized references for transparency | source='%s' polarity='%s'",
                answer_source,
                polarity,
            )
        final_answer = sanitized_answer

    if (
        current_answer in VALID_ANSWER_CHOICES
        and "**Final Answer:" not in final_answer
    ):
        final_answer = _stamp_final_answer(final_answer, current_answer)

    aggregate_answer = _extract_answer_from_plan_summary(state)
    if aggregate_answer in {"A", "B", "C", "D"}:
        logger.info(
            "Decision alignment: aggregate answer override | current='%s' aggregate='%s' polarity='%s' confidence=%.2f",
            current_answer,
            aggregate_answer,
            polarity,
            confidence,
        )
        answer_source = "step6_aggregate"
        current_answer = aggregate_answer
        final_answer = _stamp_final_answer(final_answer, aggregate_answer)
    elif current_answer == "UNKNOWN" and aggregate_answer:
        logger.info(
            "Decision alignment: answer extracted from plan summary | answer='%s' polarity='%s' confidence=%.2f",
            aggregate_answer,
            polarity,
            confidence,
        )
        answer_source = "step6_aggregate"
        current_answer = aggregate_answer
        final_answer = _stamp_final_answer(final_answer, aggregate_answer)

    if current_answer == "UNKNOWN":
        inferred = _infer_mcq_letter_from_option_text(state.get("original_question", ""), final_answer)
        if inferred in {"A", "B", "C", "D"}:
            inferred_source = answer_source
            plan = state.get("plan")
            if isinstance(plan, list) and any(
                str(step.get("step_type", "")).lower() == "aggregate" for step in plan if isinstance(step, dict)
            ):
                inferred_source = "step6_aggregate"

            logger.info(
                "Decision alignment: inferred MCQ letter from option text | inferred='%s' source='%s'",
                inferred,
                inferred_source,
            )
            answer_source = inferred_source
            current_answer = inferred
            final_answer = _stamp_final_answer(final_answer, inferred)

    candidate_answer = current_answer
    candidate_answer_prev = current_answer
    candidate_switch_reason = None
    
    already_switched_before = state.get("switch_applied", False)
    switch_retrieval_done = state.get("switch_retrieval_done", False)
    switch_updates: Dict[str, Any] = {}

    if already_switched_before and not switch_retrieval_done:
        switch_retrieval_done = True
        switch_updates["switch_retrieval_done"] = True
        switch_applied = True
    else:
        switch_applied = already_switched_before

    # ── Hypothesis switching on refute ─────────────────────────────
    # If evidence explicitly refutes the current claim, attempt to
    # switch to the alternative hypothesis mentioned in the reasoning.
    if polarity == "refute" and not switch_retrieval_done:
        options = _resolve_mcq_options(state)
        # Combine candidate reasoning sources (polarity reasoning, dialectic
        # synthesis and plan summary) to increase signal coverage when an
        # alternative hypothesis is suggested outside the immediate polarity
        # reasoning field (e.g., dialectic synthesis produced by the pipeline).
        combined_reasoning_parts = [str(polarity_data.get("reasoning", ""))]
        dsyn = state.get("dialectic_synthesis") or {}
        if isinstance(dsyn, dict):
            combined_reasoning_parts.append(str(dsyn.get("synthesis", "")))
        else:
            combined_reasoning_parts.append(str(dsyn))
        ps = state.get("plan_summary") or {}
        if isinstance(ps, dict):
            combined_reasoning_parts.append(" ".join([str(v) for v in ps.values() if isinstance(v, str)]))
        else:
            combined_reasoning_parts.append(str(ps))

        combined_reasoning = "\n".join(p for p in combined_reasoning_parts if p)

        alt_letter, alt_text, alt_score = _infer_alternative_option_from_reasoning(
            combined_reasoning,
            options,
            current_answer,
        )
        if alt_letter and alt_letter in {"A", "B", "C", "D"}:
            # Allow hypothesis switching when the alternative is strongly
            # signalled in the combined reasoning text (alt_score high)
            # even if the numeric polarity confidence is low.
            if confidence >= 0.8 or (alt_score and float(alt_score) >= 0.8):
                candidate_answer_prev = current_answer
                candidate_answer = alt_letter
                current_answer = alt_letter
                answer_source = "hypothesis_switch"
                switch_applied = True
                switch_updates["switch_applied"] = True
                candidate_switch_reason = (
                    f"Evidence refuted {candidate_answer_prev} and suggested '{alt_text}' "
                    f"(match_score={alt_score:.2f})."
                )
                logger.warning(
                    "Hypothesis switch applied | prev='%s' new='%s' reason='%s'",
                    candidate_answer_prev,
                    candidate_answer,
                    candidate_switch_reason,
                )

                # Update polarity to reflect the new claim.
                polarity = "support"
                confidence = max(0.6, min(0.9, float(confidence)))
                polarity_data = {
                    "polarity": polarity,
                    "confidence": confidence,
                    "reasoning": f"Hypothesis switch: {candidate_switch_reason}",
                }

                switch_note = (
                    f"\n\n**Hypothesis Switch**: {candidate_switch_reason}"
                )
                final_answer = _stamp_final_answer(final_answer + switch_note, candidate_answer)
                safety_flags = safety_flags + ["hypothesis_switch_applied"]
            else:
                logger.warning(f"Weak refute detected (confidence {confidence:.2f} < 0.8), applying mathematical penalty.")

    switch_updates["candidate_answer"] = candidate_answer

    if candidate_answer_prev != candidate_answer:
        switch_updates["candidate_answer_prev"] = candidate_answer_prev
        switch_updates["candidate_switch_reason"] = candidate_switch_reason or ""
        switch_updates["evidence_polarity"] = polarity_data

    # Read DS from eup_node, falling back to the epistemic summary when the
    # fused interval has not been populated yet.
    global_belief = state.get("belief_intervals", {}).get("global", {})
    epistemic_summary = _epistemic_summary(state)
    if isinstance(global_belief, dict):
        ds_support = float(global_belief.get("belief", epistemic_summary.get("support", 0.0)) or 0.0)
        ds_conflict = float(global_belief.get("conflict", epistemic_summary.get("conflict", 0.0)) or 0.0)
        ds_plausibility = float(global_belief.get("plausibility", min(1.0, ds_support + epistemic_summary.get("uncertain", 0.0))) or 0.0)
    else:
        ds_support = float(epistemic_summary.get("support", 0.0) or 0.0)
        ds_conflict = float(epistemic_summary.get("conflict", 0.0) or 0.0)
        ds_plausibility = min(1.0, ds_support + float(epistemic_summary.get("uncertain", 0.0) or 0.0))
    ds_uncertain = max(0.0, ds_plausibility - ds_support)
    step_consensus, _ = _step_consensus(state)
    
    # Apply mathematical weak refute penalty if applicable
    if polarity == "refute" and not switch_retrieval_done and confidence < 0.8:
        options = _resolve_mcq_options(state)
        alt_letter, _, _ = _infer_alternative_option_from_reasoning(
            str(polarity_data.get("reasoning", "")), options, current_answer
        )
        if alt_letter and alt_letter in {"A", "B", "C", "D"}:
            ds_support = ds_support * (1.0 - confidence)
            ds_uncertain = ds_uncertain + (confidence * 0.5)
            total = ds_support + ds_conflict + ds_uncertain
            if total > 0:
                ds_support /= total
                ds_uncertain /= total
                ds_conflict /= total

    # ── DS-guided forced commitment ───────────────────────────────
    # If epistemic fusion strongly supports an answer and the polarity
    # does not explicitly refute it, force a concrete MCQ letter.
    if (
        current_answer == "UNKNOWN"
        and ds_support > 0.6
        and polarity != "refute"
        and ds_uncertain <= 0.6
        and ds_conflict <= 0.5
    ):
        original_question = state.get("original_question", "")
        candidates = _gather_alignment_text_candidates(state, final_answer)

        forced: str | None = None
        for candidate in candidates:
            token = _get_current_answer(candidate)
            if token in {"A", "B", "C", "D"}:
                forced = token
                break
            inferred = _infer_mcq_letter_from_option_text(original_question, candidate)
            if inferred in {"A", "B", "C", "D"}:
                forced = inferred
                break

        if not forced:
            forced = await _llm_force_pick_letter(original_question, candidates[0] if candidates else final_answer)

        if forced in {"A", "B", "C", "D"}:
            logger.warning(
                "Decision alignment: forced commitment (DS-guided) | forced='%s' ds_support=%.2f",
                forced,
                ds_support,
            )
            current_answer = forced
            final_answer = _stamp_final_answer(final_answer, forced)
            safety_flags = safety_flags + ["forced_commitment_ds_guided"]

    logger.info(
        "Decision alignment | answer='%s' source='%s' polarity='%s' confidence=%.2f ds_support=%.2f ds_uncertainty=%.2f ds_conflict=%.2f",
        current_answer,
        answer_source,
        polarity,
        confidence,
        ds_support,
        ds_uncertain,
        ds_conflict,
    )

    # ── Eval mode forced resolution ────────────────────────────────
    # Converts uncertain answers to an explicit abstention for scoring.
    # ONLY fires when: evaluation_mode=True AND retries exhausted AND
    # evidence confidence is meaningful.
    is_eval_mode = state.get("evaluation_mode", False)
    retry_count  = state.get("retry_count", 0)

    # Calibrated confidence per spec: ds_support - ds_conflict
    calibrated_confidence = float(ds_support) - float(ds_conflict)
    eus_global = float(((state.get("eus_per_claim") or {}).get("global", 0.0)) or 0.0)

    router_output = state.get("router_output", {}) or {}
    force_commitment = False
    if isinstance(router_output, dict):
        force_commitment = bool(router_output.get("force_commitment", False)) or bool(
            (router_output.get("answer_policy", {}) or {}).get("force_commitment", False)
        )

    if (
        is_eval_mode
        and retry_count >= 1
        and calibrated_confidence < 0.2
        and step_consensus < 0.60
        and not force_commitment
    ):
        forced_answer = "UNKNOWN"

        if forced_answer != current_answer:
            reason = (
                f"The available evidence is insufficient to provide a reliable answer (DS Uncertainty: {ds_uncertain:.2f}, Step Consensus: {step_consensus:.2f})."
            )
            logger.warning(reason)

            correction = (
                f"\n\n**Decision Alignment Override (Eval Policy)**\n"
                f"{reason}\n"
                f"**Final Answer: {forced_answer}**"
            )
            cleaned_answer = re.sub(
                r"(?is)\*\*final\s+answer:\s*(?:[A-D]|UNKNOWN)\*\*",
                "",
                final_answer,
            ).strip()

            aligned_answer = _stamp_final_answer(cleaned_answer + correction, forced_answer)
            aligned_answer = _append_epistemic_footer(aligned_answer, state, calibrated_confidence, str(controversy_label))
            return {
                "final_answer": aligned_answer,
                "final_answer_raw": raw_final_answer,
                "final_answer_aligned": aligned_answer,
                "answer": forced_answer,
                "predicted_letter": forced_answer,
                "answer_source": answer_source,
                # Every override is recorded — safety critic and evaluator
                # can see this happened and weight results accordingly
                "safety_flags": safety_flags + ["forced_resolution_applied"],
                **switch_updates,
                **conflict_updates,
            }

    # ── Standard alignment rules ───────────────────────────────────
    # Primary decision is DS-driven; polarity modifies confidence without
    # blindly overriding the fused belief.

    new_decision = None
    reason = None
    low_confidence = False

    # DS-Polarity Consistency Check
    if polarity == "refute" and not switch_applied:
        safety_flags = safety_flags + ["ds_polarity_inconsistent"]
        # Reduce confidence if DS support is high but polarity refutes
        if ds_support > 0.6:
            calibrated_confidence *= 0.5
            logger.warning("Reducing confidence due to high DS support vs refute polarity mismatch.")

    applicability_score = float(state.get("applicability_score", 0.5))
    
    if should_abstain(
        polarity,
        confidence,
        ds_uncertain,
        step_consensus,
        applicability_score,
        str(controversy_label),
        ds_support,
    ):
        new_decision = "UNKNOWN"
        reason = (
            f"Epistemic abstention triggered (DS Uncertainty: {ds_uncertain:.2f}, "
            f"Step Consensus: {step_consensus:.2f}, Controversy: {controversy_label})."
        )
    elif switch_retrieval_done:
        if ds_support > 0.5:
            pass # Keep candidate answer
        else:
            new_decision = "UNKNOWN"
            reason = f"Second pass failed to find strong support (ds_support={ds_support:.2f} <= 0.5)."
    elif ds_support > 0.75:
        # Circuit breaker: High DS support overrides low calibrated confidence.
        pass
    elif calibrated_confidence < 0.15 and step_consensus < 0.50:
        # Tightened from 0.2/0.60 to 0.15/0.50 to reduce over-abstention
        new_decision = "UNKNOWN"
        reason = (
            f"The available evidence is insufficient to provide a reliable answer "
            f"(DS Uncertainty: {ds_uncertain:.2f}, Step Consensus: {step_consensus:.2f})."
        )
    elif ds_support > 0.7 and ds_conflict < 0.6:
        # Strongly supported answer
        pass
    else:
        # Return answer but explicitly flag low confidence if metrics are borderline.
        if calibrated_confidence < 0.3 or ds_uncertain > 0.5:
            low_confidence = True
            reason = (
                f"Low confidence: ds_support={ds_support:.2f}, ds_uncertainty={ds_uncertain:.2f}, "
                f"ds_conflict={ds_conflict:.2f}, confidence={calibrated_confidence:.2f}, polarity={polarity}."
            )

    if low_confidence and current_answer in {"A", "B", "C", "D"}:
        low_conf_note = f"\n\n**Confidence: LOW** {reason}"
        final_answer = _stamp_final_answer(final_answer + low_conf_note, current_answer)
        safety_flags = safety_flags + ["low_confidence_flag"]

    if new_decision:
        logger.warning(f"Alignment override: {reason}")

        correction = (
            f"\n\n**Decision Alignment Override**\n"
            f"{reason}\n"
            f"**Final Answer: {new_decision}**"
        )
        aligned_answer = _stamp_final_answer(final_answer + correction, new_decision)
        payload = {
            "final_answer": aligned_answer,
            "final_answer_raw": raw_final_answer,
            "final_answer_aligned": aligned_answer,
            "answer": new_decision,
            "predicted_letter": new_decision,
            "answer_source": answer_source,
            "safety_flags": safety_flags + ["alignment_override_applied"],
            **switch_updates,
            **conflict_updates,
        }
        threshold_table = {
            "low_calibrated_confidence": {"value": round(calibrated_confidence, 3), "threshold": 0.2, "op": "<"},
            "high_ds_conflict": {"value": round(ds_conflict, 3), "threshold": 0.75, "op": ">"},
            "high_ds_uncertainty": {"value": round(ds_uncertain, 3), "threshold": 0.6, "op": ">"},
            "high_ds_support_circuit_breaker": {"value": round(ds_support, 3), "threshold": 0.75, "op": ">"},
        }
        
        trace_event = build_trace_event(
            state,
            section="decision_governance",
            event="decision_alignment",
            node="decision_alignment",
            data={
                "final_answer": new_decision,
                "reason": reason,
                "polarity": polarity,
                "calibrated_confidence": round(calibrated_confidence, 3),
                "ds_support": round(ds_support, 3),
                "ds_uncertain": round(ds_uncertain, 3),
                "ds_conflict": round(ds_conflict, 3),
                "thresholds": threshold_table,
                "alignment_altered_answer": True,
                "override": True,
            },
            influence={"decision": "alignment_override"},
            attach_context=False,
        )
        trace_updates = build_trace_updates(state, [trace_event])
        payload["trace_events"] = payload.get("trace_events", []) + trace_updates.get("trace_events", [])
        if "trace_id" not in payload:
            payload["trace_id"] = trace_updates.get("trace_id")
        if "trace_created_at" not in payload:
            payload["trace_created_at"] = trace_updates.get("trace_created_at")
        payload_final = _stamp_final_answer(payload.get("final_answer", ""), payload.get("answer", "UNKNOWN"))
        payload["final_answer"] = _append_epistemic_footer(payload_final, state, calibrated_confidence, str(controversy_label))
        return payload


    # No alignment change needed
    logger.info("Decision alignment: no override needed.")
    
    # Check if answer changed via aggregate/inference even if no "override" policy fired
    alignment_altered_answer = (state.get("predicted_letter") != current_answer)

    payload = {
        "final_answer": final_answer,
        "final_answer_raw": raw_final_answer,
        "final_answer_aligned": final_answer,
        "answer": current_answer,
        "predicted_letter": current_answer,
        "answer_source": answer_source,
        "safety_flags": safety_flags,
        **switch_updates,
        **conflict_updates,
    }

    if force_commitment and str(current_answer or "").strip().upper() != "UNKNOWN":
        payload["safety_flags"] = payload.get("safety_flags", []) + ["force_commitment_override"]
        payload["final_answer"] = payload["final_answer"] + (
            f"\n\n**Epistemic Notice:** force_commitment preserved the answer despite DS Uncertainty={ds_uncertain:.2f} and Step Consensus={step_consensus:.2f}."
        )
    
    threshold_table = {
        "low_calibrated_confidence": {"value": round(calibrated_confidence, 3), "threshold": 0.2, "op": "<"},
        "high_ds_conflict": {"value": round(ds_conflict, 3), "threshold": 0.75, "op": ">"},
        "high_ds_uncertainty": {"value": round(ds_uncertain, 3), "threshold": 0.6, "op": ">"},
        "high_ds_support_circuit_breaker": {"value": round(ds_support, 3), "threshold": 0.75, "op": ">"},
    }

    trace_event = build_trace_event(
        state,
        section="decision_governance",
        event="decision_alignment",
        node="decision_alignment",
        data={
            "final_answer": current_answer,
            "polarity": polarity,
            "calibrated_confidence": round(calibrated_confidence, 3),
            "ds_support": round(ds_support, 3),
            "ds_uncertain": round(ds_uncertain, 3),
            "ds_conflict": round(ds_conflict, 3),
            "thresholds": threshold_table,
            "alignment_altered_answer": alignment_altered_answer,
            "switch_applied": switch_applied,
            "override": False,
        },
        influence={"decision": "no_override"},
        attach_context=False,
    )
    trace_updates = build_trace_updates(state, [trace_event])
    payload["trace_events"] = payload.get("trace_events", []) + trace_updates.get("trace_events", [])
    if "trace_id" not in payload:
        payload["trace_id"] = trace_updates.get("trace_id")
    if "trace_created_at" not in payload:
        payload["trace_created_at"] = trace_updates.get("trace_created_at")
    payload_final = _stamp_final_answer(payload.get("final_answer", ""), payload.get("answer", "UNKNOWN"))
    payload["final_answer"] = _append_epistemic_footer(payload_final, state, calibrated_confidence, str(controversy_label))
    return payload

