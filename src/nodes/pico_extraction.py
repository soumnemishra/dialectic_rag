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
        if not isinstance(pico_res, PICO):
            if isinstance(pico_res, dict):
                pico_res = PICO(**pico_res)
            else:
                pico_res = PICO(population="unknown", intervention="unknown", outcome="unknown", risk_level="unknown")
        
        # Add trace event
        trace_event = {
            "node": "pico_extraction",
            "section": "intention_classification",
            "input": {"vignette_length": len(vignette)},
            "output": {"pico": pico_res, "candidate_answers": candidate_answers},
            "risk_level": pico_res.get("risk_level", "unknown")
        }
        
        return {
            "pico": pico_res,
            "risk_level": pico_res.get("risk_level", "unknown"),
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
