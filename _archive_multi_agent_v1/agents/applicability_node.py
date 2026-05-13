import logging
import os
from typing import Any, Dict, List

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.core.registry import ModelRegistry, safe_ainvoke
from src.prompts.templates import with_json_system_suffix
from src.query_builder import parse_markdown_json
from src.state.state import GraphState
from src.utils.epistemic_trace import build_trace_event, build_trace_updates

logger = logging.getLogger(__name__)


APPLICABILITY_SYSTEM_PROMPT = (
    "You are a patient-cohort alignment auditor for a medical RAG system. "
    "Compare the patient demographics in the query against the study populations mentioned in the evidence. "
    "Focus on age, sex, pregnancy status, geography, care setting, disease stage, and major comorbidities. "
    "Return only JSON."
)

APPLICABILITY_HUMAN_PROMPT = """Patient vignette:
{question}

Evidence items:
{evidence}

For each evidence item, output a cohort_match_score between 0.0 and 1.0 where 0.0 means totally different populations (for example pediatric vs geriatric) and 1.0 means an exact demographic match.

Return a JSON object with this shape:
{{
  "cohort_match_scores": [
    {{
      "evidence_index": 1,
      "cohort_match_score": 0.82,
      "reasoning": "Brief rationale"
    }}
  ]
}}
"""

EVIDENCE_CHAR_LIMIT = 12000
TRUNCATION_SUFFIX = "... [TRUNCATED]"


def _aggregate_scores(scores: List[float]) -> float:
    mode = os.getenv("MRAGE_APPLICABILITY_AGG", "max").strip().lower()
    if mode in {"mean", "avg", "average"}:
        return sum(scores) / len(scores)
    if mode == "median":
        ordered = sorted(scores)
        mid = len(ordered) // 2
        if len(ordered) % 2 == 1:
            return ordered[mid]
        return (ordered[mid - 1] + ordered[mid]) / 2
    if mode == "min":
        return min(scores)
    if mode in {"p75", "percentile_75"}:
        ordered = sorted(scores)
        idx = int(round(0.75 * (len(ordered) - 1)))
        return ordered[max(0, min(len(ordered) - 1, idx))]
    return max(scores)


class CohortMatchItem(BaseModel):
    evidence_index: int = Field(default=0)
    cohort_match_score: float = Field(default=0.5, ge=0.0, le=1.0)
    reasoning: str = Field(default="")


class ApplicabilityOutput(BaseModel):
    cohort_match_scores: List[CohortMatchItem] = Field(default_factory=list)


def _format_question_and_evidence(question: str, step_notes: List[Any]) -> str:
    lines: List[str] = []
    for idx, note in enumerate(step_notes, start=1):
        note_text = str(note).strip()
        if not note_text:
            continue
        lines.append(f"Evidence {idx}:\n{note_text}")
    return "\n\n".join(lines) if lines else "No evidence provided."


async def applicability_scoring_node(state: GraphState) -> Dict[str, Any]:
    """
    Compare the patient vignette against the study populations mentioned in the retrieved evidence.

    Reads:  state["original_question"], state["step_notes"]
    Writes: state["applicability_score"]
    """
    question = str(state.get("original_question", "")).strip()
    step_notes = state.get("step_notes", []) or []

    if not question or not step_notes:
        payload = {"applicability_score": 0.5}
        trace_event = build_trace_event(
            state,
            section="evidence_analysis",
            event="applicability_scoring",
            node="applicability_scoring",
            data={
                "applicability_score": 0.5,
                "evidence_items": len(step_notes),
                "note": "missing_question_or_evidence",
            },
            influence={"state_updates": ["applicability_score"]},
            attach_context=False,
        )
        trace_updates = build_trace_updates(state, [trace_event])
        payload["trace_events"] = payload.get("trace_events", []) + trace_updates.get("trace_events", [])
        if "trace_id" not in payload:
            payload["trace_id"] = trace_updates.get("trace_id")
        if "trace_created_at" not in payload:
            payload["trace_created_at"] = trace_updates.get("trace_created_at")
        return payload

    evidence_text = _format_question_and_evidence(question, step_notes)
    if len(evidence_text) > EVIDENCE_CHAR_LIMIT:
        safe_evidence = evidence_text[:EVIDENCE_CHAR_LIMIT] + TRUNCATION_SUFFIX
    else:
        safe_evidence = evidence_text

    try:
        llm = ModelRegistry.get_light_llm(temperature=0.0, json_mode=True)
        if llm is None:
            raise RuntimeError("Light LLM unavailable")

        parser = JsonOutputParser(pydantic_object=ApplicabilityOutput)
        prompt = ChatPromptTemplate.from_messages([
            ("system", with_json_system_suffix(APPLICABILITY_SYSTEM_PROMPT)),
            ("human", APPLICABILITY_HUMAN_PROMPT),
        ])

        chain = prompt | llm | parser
        result = await safe_ainvoke(chain, {"question": question, "evidence": safe_evidence})

        if isinstance(result, dict):
            raw_items = result.get("cohort_match_scores", [])
        else:
            raw_items = getattr(result, "cohort_match_scores", [])

        scores: List[float] = []
        for item in raw_items or []:
            if isinstance(item, dict):
                try:
                    score = float(item.get("cohort_match_score", 0.5))
                    scores.append(score)
                except (TypeError, ValueError):
                    continue
            elif hasattr(item, "cohort_match_score"):
                try:
                    score = float(getattr(item, "cohort_match_score", 0.5))
                    scores.append(score)
                except (TypeError, ValueError):
                    continue

        if not scores:
            fallback = result.get("cohort_match_score") if isinstance(result, dict) else None
            if fallback is not None:
                try:
                    scores.append(float(fallback))
                except (TypeError, ValueError):
                    pass

        if not scores:
            raise ValueError("No applicability scores returned")

        applicability_score = round(_aggregate_scores(scores), 3)
        logger.info(
            "Applicability scoring complete | items=%d agg=%s score=%.3f",
            len(scores),
            os.getenv("MRAGE_APPLICABILITY_AGG", "max"),
            applicability_score,
        )
        payload = {"applicability_score": applicability_score}
        trace_event = build_trace_event(
            state,
            section="evidence_analysis",
            event="applicability_scoring",
            node="applicability_scoring",
            data={
                "applicability_score": applicability_score,
                "evidence_items": len(step_notes),
            },
            influence={"state_updates": ["applicability_score"]},
            attach_context=False,
        )
        trace_updates = build_trace_updates(state, [trace_event])
        payload["trace_events"] = payload.get("trace_events", []) + trace_updates.get("trace_events", [])
        if "trace_id" not in payload:
            payload["trace_id"] = trace_updates.get("trace_id")
        if "trace_created_at" not in payload:
            payload["trace_created_at"] = trace_updates.get("trace_created_at")
        return payload

    except Exception as exc:
        logger.warning("Applicability scoring failed; using neutral fallback: %s", exc)
        payload = {"applicability_score": 0.5}
        trace_event = build_trace_event(
            state,
            section="evidence_analysis",
            event="applicability_scoring",
            node="applicability_scoring",
            data={
                "applicability_score": 0.5,
                "evidence_items": len(step_notes),
                "note": "fallback",
            },
            influence={"state_updates": ["applicability_score"]},
            attach_context=False,
        )
        trace_updates = build_trace_updates(state, [trace_event])
        payload["trace_events"] = payload.get("trace_events", []) + trace_updates.get("trace_events", [])
        if "trace_id" not in payload:
            payload["trace_id"] = trace_updates.get("trace_id")
        if "trace_created_at" not in payload:
            payload["trace_created_at"] = trace_updates.get("trace_created_at")
        return payload