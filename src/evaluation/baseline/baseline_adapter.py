"""Adapter that wraps the vanilla RAG path into the shared evaluation interface."""

from __future__ import annotations

import logging
from typing import Any, Dict

from src.agents.rag import RagAgent
from src.evaluation.adapter import AgentResult

logger = logging.getLogger(__name__)


class BaselineAdapter:
    """Wrap the vanilla RAG system into the shared AgentResult interface."""

    def __init__(self, baseline_rag: RagAgent | None = None):
        self.rag = baseline_rag or RagAgent()

    async def answer_query(self, prompt: str) -> AgentResult:
        state: Dict[str, Any] = {
            "question": prompt,
            "documents": [],
            "doc_ids": [],
            "notes": [],
            "final_raw_answer": {},
            "intent": "informational",
            "risk_level": "low",
            "safety_flags": [],
            "evidence_polarity": {
                "polarity": "insufficient",
                "confidence": 0.0,
                "reasoning": "baseline path",
            },
        }

        try:
            result = await self.rag.query(state)
            final_raw = result.get("final_raw_answer", {})
            answer = ""
            if isinstance(final_raw, dict):
                answer = str(final_raw.get("answer", ""))
            if not answer:
                answer = str(result.get("final_answer", ""))
            if not answer:
                answer = "No answer generated."

            sources = result.get("doc_ids", [])
            metadata = {
                "system": "baseline_rag",
                "answer_source": "baseline_rag",
                "tcs_score": None,
                "rps_avg": None,
                "controversy_label": None,
                "eus": None,
                "risk_level": "unknown",
                "safety_flags": result.get("safety_flags", []),
                "safety_intercepted": False,
            }
            return AgentResult(answer=answer, sources=list(sources), metadata=metadata)
        except Exception as exc:
            logger.error("Baseline adapter failed: %s", exc, exc_info=True)
            return AgentResult(
                answer=f"Error: {exc}",
                sources=[],
                metadata={"system": "baseline_rag", "answer_source": "baseline_rag"},
            )
