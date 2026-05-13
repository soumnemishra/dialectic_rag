#!/usr/bin/env python3
"""Run benchmark question 0228 through the full workflow and print the conflict-analysis outputs."""

import asyncio
import json
import logging
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import configure_logging
from src.graph.workflow import build_workflow
from src.models.state import GraphState

configure_logging()
logger = logging.getLogger(__name__)


def load_question_0228() -> tuple[str, dict[str, str], str]:
    benchmark_path = Path(__file__).resolve().parents[1] / "data" / "benchmark.json"
    data = json.loads(benchmark_path.read_text(encoding="utf-8"))
    item = data["medqa"]["0228"]
    question = item["question"]
    options = item["options"]
    answer = item.get("answer", "")
    return question, options, answer


async def main() -> None:
    question, options, answer = load_question_0228()
    options_text = "\n".join(f"{key}: {value}" for key, value in options.items())
    prompt = f"{question}\n\nOptions:\n{options_text}"

    workflow = build_workflow()
    initial_state: GraphState = {
        "original_question": prompt,
        "mcq_options": None,
        "intent": "clinical_question",
        "risk_level": "moderate",
        "pico": None,
        "evidence_pool": [],
        "retrieved_docs": {},
        "step_notes": [],
        "claim_clusters": [],
        "temporal_result": None,
        "consensus_state": None,
        "epistemic_result": None,
        "abstention_rationale": None,
        "candidate_answers": [],
        "candidate_answer": "",
        "final_reasoning": "",
        "abstention_triggered": False,
        "extracted_claims": [],
        "candidate_stances": {},
        "fused_beliefs": {},
        "temporal_shift": {},
        "epistemic_state": None,
        "safety_flags": [],
        "trace_events": [],
        "trace_id": str(uuid.uuid4()),
        "trace_created_at": datetime.utcnow().isoformat(),
    }

    logger.info("Running question 0228; gold answer=%s", answer)
    result = await workflow.ainvoke(initial_state)

    summary = {
        "candidate_stances": result.get("candidate_stances"),
        "temporal_shift": result.get("temporal_shift"),
        "epistemic_state": result.get("epistemic_state"),
        "consensus_state": result.get("consensus_state"),
        "claim_clusters": len(result.get("claim_clusters", []) or []),
        "extracted_claims": len(result.get("extracted_claims", []) or []),
        "evidence_pool": len(result.get("evidence_pool", []) or []),
        "trace_events": len(result.get("trace_events", []) or []),
    }
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
