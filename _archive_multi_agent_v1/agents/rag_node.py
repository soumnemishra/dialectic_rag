import logging
import re
from typing import Dict, Any, List

from src.state.state import GraphState, RagState
from src.utils.epistemic_trace import build_trace_event, build_trace_updates

logger = logging.getLogger(__name__)

_FINAL_ANSWER_RE = re.compile(
    r"(?is)(?:\*\*\s*)?final\s*answer\s*(?:\*\*\s*)?[:\-\s]*([A-D])\b"
)

def _extract_final_answer_letter(text: str) -> str | None:
    if not text:
        return None
    match = _FINAL_ANSWER_RE.search(str(text))
    if match:
        letter = match.group(1).upper()
        if letter in {"A", "B", "C", "D"}:
            return letter
    return None

async def rag_direct_node(state: GraphState) -> Dict[str, Any]:
    """
    Entry retrieval and initial synthesis module.
    Performs primary retrieval and generates the first-pass answer.
    """
    logger.info("Executing Primary Retrieval (rag_direct_node)...")

    question = state["original_question"]
    mcq_options = state.get("mcq_options", "")

    # ── Build RagState ─────────────────────────────
    # We pass empty documents so RagAgent performs fresh retrieval
    rag_input: RagState = {
        "question":         question,
        "documents":        [],
        "doc_ids":          [],
        "notes":            [],
        "final_raw_answer": {},
        "intent":           state.get("intent",      "informational"),
        "risk_level":       state.get("risk_level",  "low"),
        "safety_flags":     state.get("safety_flags", []),
        "evidence_polarity": state.get("evidence_polarity", {}),
        "original_question": question,
        "mcq_options":      mcq_options,
    }

    try:
        from src.agents.registry import AgentRegistry
        agent = AgentRegistry.get_instance().rag
        result = await agent.query(rag_input)

        final_raw  = result.get("final_raw_answer", {})
        predicted_letter = str(final_raw.get("predicted_letter", "UNKNOWN")).upper()
        answer_text = str(final_raw.get("answer", predicted_letter))
        final_answer_text = str(final_raw.get("final_answer", "")).strip()
        
        if not final_answer_text:
            final_answer_text = f"{final_raw.get('clinical_reasoning', 'Initial synthesis completed.')}\n\n**Final Answer: {predicted_letter}**"
        
        doc_ids = result.get("doc_ids", [])
        step_notes_out = result.get("notes", [])

        # Build a step_output entry for downstream epistemic modules
        step_output_entry = {
            "analysis": final_raw.get("clinical_reasoning", "Initial synthesis"),
            "answer":   answer_text,
            "predicted_letter": predicted_letter,
            "success":  final_raw.get("success", "Yes"),
            "rating":   final_raw.get("rating",  8),
            "is_error": False,
        }

        payload = {
            "final_answer":  final_answer_text,
            "predicted_letter": predicted_letter,
            "step_output":   [step_output_entry],
            "step_docs_ids": [doc_ids],
            "step_notes":    step_notes_out,
        }

        trace_event = build_trace_event(
            state,
            section="retrieval",
            event="rag_direct",
            node="rag_direct",
            data={
                "doc_count": len(doc_ids),
                "predicted_letter": predicted_letter,
            },
            influence={"state_updates": ["final_answer", "step_docs_ids", "step_output"]},
            attach_context=False,
        )
        trace_updates = build_trace_updates(state, [trace_event])
        payload.update(trace_updates)
        return payload

    except Exception as e:
        logger.error(f"rag_direct_node failed: {e}", exc_info=True)
        return {
            "final_answer":  "I encountered an error during primary retrieval.",
            "predicted_letter": "UNKNOWN",
            "step_output":   [],
            "step_docs_ids": [],
            "step_notes":    [],
        }