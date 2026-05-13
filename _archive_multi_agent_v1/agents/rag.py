"""Legacy-compatible `RagAgent` implementation.

This module provides a minimal `RagAgent` class expected by
`src.agents.registry.AgentRegistry` and by legacy adapters.

The implementation is intentionally lightweight: it uses the
configured `RetrieverTool` to run a PubMed search and returns a
simple synthesis payload. This is sufficient for evaluation and
end-to-end smoke tests and avoids reintroducing the old complex
RAG orchestration.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.pubmed_client import RetrievedDocument

logger = logging.getLogger(__name__)


class RagAgent:
	"""Lightweight RAG agent used as a compatibility shim.

	Methods:
		- query(state): async method taking a RagState-like dict and
		  returning a dict with keys expected by downstream modules.
	"""

	def __init__(self, retriever_tool: Any | None = None):
		# Lazy import to avoid circular init problems
		from src.agents.registry import AgentRegistry

		self._registry = AgentRegistry.get_instance()
		self.retriever = retriever_tool or getattr(self._registry, "retriever", None)

	async def query(self, state: Dict[str, Any]) -> Dict[str, Any]:
		"""Perform a simple retrieval + lightweight synthesis.

		Returns a dict containing at least:
			- final_raw_answer: dict with `predicted_letter` and `answer`
			- final_answer: readable final answer text
			- doc_ids: list of PMIDs used
			- notes: list of textual notes
			- safety_flags: list
		"""
		question = state.get("question") or state.get("original_question") or ""
		try:
			# Use the RetrieverTool client which exposes `search`
			if self.retriever and getattr(self.retriever, "client", None):
				docs = await self.retriever.client.search(question, max_results=3)
			else:
				docs = []

			doc_ids: List[str] = [d.pmid for d in docs if hasattr(d, "pmid")]

			# Lightweight synthesized answer: use first doc title + PMID
			if docs:
				first: RetrievedDocument = docs[0]
				final_text = f"{first.to_context_string()}\n\n**Final Answer: UNKNOWN**"
				predicted = "UNKNOWN"
				notes = [f"Retrieved {len(docs)} documents; top PMID={first.pmid}"]
			else:
				final_text = "No retrieved evidence." + "\n\n**Final Answer: UNKNOWN**"
				predicted = "UNKNOWN"
				notes = ["No documents retrieved."]

			return {
				"final_raw_answer": {
					"predicted_letter": predicted,
					"answer": predicted,
					"clinical_reasoning": "Lightweight retrieval synthesis",
					"success": True,
				},
				"final_answer": final_text,
				"doc_ids": doc_ids,
				"notes": notes,
				"safety_flags": [],
			}

		except Exception as e:
			logger.exception("RagAgent.query failed: %s", e)
			return {
				"final_raw_answer": {"predicted_letter": "UNKNOWN", "answer": "UNKNOWN", "clinical_reasoning": "error"},
				"final_answer": "Error during retrieval.",
				"doc_ids": [],
				"notes": [str(e)],
				"safety_flags": ["retrieval_error"],
			}


__all__ = ["RagAgent"]
