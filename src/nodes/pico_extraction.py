import logging
from typing import Dict, Any
from src.models.state import GraphState
from src.models.schemas import PICO
from src.retrieval.pico_extractor import PICOExtractor

logger = logging.getLogger(__name__)

async def pico_extraction_node(state: GraphState) -> Dict[str, Any]:
    # 1. Strip MCQ options for Question-Only Retrieval (QOR)
    question = state.get("original_question", "")
    import re
    # Match standard delimiters and take the first part (the vignette)
    parts = re.split(r"(?:\n\s*Options:\s*\n|\n\s*A:|\n\s*A\))", question, maxsplit=1)
    vignette = parts[0].strip()
    
    # Extract candidate answers from the options part
    candidate_answers = []
    if len(parts) > 1:
        # Simplistic parsing of "A) option A\nB) option B"...
        options_text = question[len(vignette):]
        # Find lines like "A) ..." or "A: ..."
        lines = options_text.split('\n')
        for line in lines:
            line = line.strip()
            if re.match(r"^[A-E][:\)]", line):
                candidate_answers.append(line.split(maxsplit=1)[1].strip())
    
    # Fallback to default if empty
    if not candidate_answers:
        candidate_answers = ["Intervention is supported", "Intervention is not supported"]

    extractor = PICOExtractor()
    try:
        pico_res = await extractor.extract(vignette)

        # Normalize LLM-wrapped responses: some LLM adapters return a dict
        # with a 'content' field containing the JSON string (or fenced```json).
        if isinstance(pico_res, dict) and "content" in pico_res:
            raw = pico_res.get("content") or ""
            raw = raw.strip()
            
            # Strip code fences (e.g., ```json ... ``` or ``` ... ```)
            if raw.startswith("```"):
                # Find and remove opening fence line
                lines = raw.split("\n")
                lines = [l for l in lines if l.strip()]  # Remove empty lines
                # Remove first line (opening fence)
                if lines and lines[0].strip().startswith("```"):
                    lines = lines[1:]
                # Remove last line (closing fence) if it's just backticks
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                raw = "\n".join(lines).strip()

            # Try to parse JSON content if possible
            try:
                import json
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    pico_res = parsed
            except Exception:
                pass

        # Ensure we end up with a PICO instance or a safe fallback
        if not isinstance(pico_res, PICO):
            if isinstance(pico_res, dict):
                try:
                    pico_res = PICO(**pico_res)
                except Exception as e:
                    logger.warning("PICO validation failed; falling back to defaults: %s", e)
                    pico_res = PICO(intent="Therapeutic", population="unknown", intervention="unknown", outcome="unknown", risk_level="unknown")
            else:
                pico_res = PICO(intent="Therapeutic", population="unknown", intervention="unknown", outcome="unknown", risk_level="unknown")

        # Normalize pico for trace and return (always a dict)
        pico_dict = pico_res.dict() if isinstance(pico_res, PICO) else (pico_res or {})

        # Add trace event
        trace_event = {
            "node": "pico_extraction",
            "section": "intention_classification",
            "input": {"vignette_length": len(vignette)},
            "output": {
                "intent": getattr(pico_res, "intent", None),
                "pico": pico_dict,
                "candidate_answers": candidate_answers
            },
            "evaluation_policy": {
                "zero_shot": True,
                "question_only_retrieval": True,
                "options_visible_to_retrieval": False,
                "mcq_options_present_but_hidden": bool(state.get("mcq_options")),
            },
            "risk_level": pico_dict.get("risk_level", "unknown")
        }

        return {
            "pico": pico_dict,
            "risk_level": pico_dict.get("risk_level", "unknown"),
            "candidate_answers": candidate_answers,
            "trace_events": [trace_event]
        }
    except Exception as e:
        logger.error(f"PICO extraction node failed: {e}")
        # Fallback is handled within PICOExtractor.extract
        default_res = {"population": "unknown", "intervention": "unknown", "comparator": "unknown", "outcome": "unknown", "risk_level": "unknown"}
        return {
            "pico": default_res,
            "risk_level": "unknown",
            "candidate_answers": candidate_answers,
            "trace_events": [{"node": "pico_extraction", "error": str(e)}]
        }
