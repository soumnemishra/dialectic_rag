"""Dialectical / contrastive retriever wrapper.

Performs a normal retrieval plus a set of intent-specific adversarial queries and
merges results giving a boost to documents returned by the adversarial queries.

The wrapper preserves the RetrieverTool API and is intended to be toggled via
experimental flags.
"""

from typing import List, Optional
import logging

from src.config import settings
from src.query_builder import IntentQueryStrategy

logger = logging.getLogger(__name__)


class DialecticalRetriever:
    def __init__(
        self,
        base_retriever,
        boost_factor: float = 2.0,
        contrastive_terms: List[str] | None = None,
        intent: str = "informational"
    ):
        self.base = base_retriever
        self.boost = float(boost_factor)
        self.intent = intent
        
        # Use intent-specific adversarial suffixes; fall back to generic terms
        self.terms = contrastive_terms or IntentQueryStrategy.get_adversarial_suffixes(intent)
        if not self.terms:
            # Ultimate fallback for unknown intents
            self.terms = [
                "however",
                "contradicts",
                "contrary to",
                "inconsistent",
                "no significant",
            ]
        
        logger.info(
            "DialecticalRetriever initialized",
            extra={
                "intent": intent,
                "adversarial_suffixes": len(self.terms),
                "boost_factor": boost_factor
            }
        )

    async def retrieve(self, query: str, top_k: int = 10, intent: Optional[str] = None, **kwargs):
        """
        Retrieve supporting evidence and dialectical (contradictory) evidence.
        
        Args:
            query: The search query
            top_k: Number of results to return
            intent: Clinical intent to guide adversarial query generation
            **kwargs: Additional arguments to pass to base retriever
        
        Returns:
            Tuple of (merged_docs, merged_doc_ids)

        Side effects:
            Sets ``self.last_metadata`` and ``self.last_opposing_evidence_found``
            so downstream epistemic nodes can inspect the dialectical search
            outcome without changing legacy call sites.
        """
        # Update intent if provided
        if intent:
            self.intent = intent
            self.terms = IntentQueryStrategy.get_adversarial_suffixes(intent)
        
        # Run baseline retrieval
        try:
            base_docs, base_doc_ids = await self.base(query, **kwargs)
        except Exception as e:
            logger.warning(f"Base retrieval failed: {e}")
            base_docs, base_doc_ids = [], []

        # Run adversarial/dialectical retrievals
        dialectical_docs = []
        dialectical_doc_ids = []
        self.last_opposing_evidence_found = False
        self.last_metadata = {
            "base_docs": len(base_docs),
            "dialectical_docs": 0,
            "dialectical_search_executed": False,
            "dialectical_zero_hit": True,
        }

        try:
            # Lazy import to avoid startup circular imports
            from src.core.registry import ModelRegistry, safe_ainvoke
            from langchain_core.prompts import ChatPromptTemplate
            from src.prompts.templates import (
                ADVERSARIAL_QUERY_SYSTEM_PROMPT,
                ADVERSARIAL_QUERY_HUMAN_PROMPT,
                with_json_system_suffix,
            )

            llm = ModelRegistry.get_flash_llm(temperature=0.0, json_mode=True)
            if llm is not None:
                # Try to generate intent-specific adversarial query
                intent_for_llm = self.intent if self.intent != "informational" else "treatment"
                
                prompt = ChatPromptTemplate.from_messages([
                    ("system", with_json_system_suffix(ADVERSARIAL_QUERY_SYSTEM_PROMPT)),
                    ("human", ADVERSARIAL_QUERY_HUMAN_PROMPT),
                ])

                payload = {
                    "original_query": query,
                    "intent": intent_for_llm,
                    "current_hypothesis": query
                }
                
                try:
                    result = await safe_ainvoke(prompt | llm, payload)
                    adversarial_query = str(result.get("adversarial_query", "")).strip()
                    logger.info(
                        "LLM adversarial query generated",
                        extra={
                            "intent": intent_for_llm,
                            "original_query": query[:60],
                            "adversarial_query": adversarial_query[:60]
                        }
                    )
                except Exception as e:
                    logger.debug(f"LLM adversarial query generation failed: {e}")
                    adversarial_query = ""
            else:
                adversarial_query = ""
        except Exception as e:
            logger.debug(f"LLM adversarial query setup failed: {e}")
            adversarial_query = ""

        # QUALITY GATE: if the adversarial query is too long (echoing the full question), fall back
        if adversarial_query and len(adversarial_query.split()) > 10:
            logger.warning(
                "LLM adversarial query too long (likely echo); falling back to suffix-based",
                extra={"adversarial_query": adversarial_query[:80]}
            )
            adversarial_query = ""

        # Stage 2: Execute adversarial sub-queries
        if adversarial_query:
            try:
                sub_kwargs = dict(kwargs)
                sub_kwargs["skip_query_builder"] = True
                docs, ids = await self.base(adversarial_query, **sub_kwargs)
                if docs:
                    dialectical_docs.extend(docs)
                    dialectical_doc_ids.extend(ids)
                    self.last_opposing_evidence_found = True
                    self.last_metadata = {
                        "base_docs": len(base_docs),
                        "dialectical_docs": len(dialectical_docs),
                        "dialectical_search_executed": True,
                        "dialectical_zero_hit": len(dialectical_docs) == 0,
                    }
                    logger.info(
                        "Adversarial query succeeded",
                        extra={"adversarial_query": adversarial_query[:60], "results": len(docs)}
                    )
                else:
                    logger.warning("Adversarial query yielded zero documents; triggering suffix fallback")
                    adversarial_query = ""  # Trigger suffix fallback below
            except Exception as e:
                logger.debug(f"Dialectical adversarial subquery failed: {e}")
                adversarial_query = ""

        # Fallback: Use intent-specific suffix-based queries
        if not adversarial_query:
            terms_to_use = self.terms
            if settings.FAST_EPISTEMIC:
                # In fast mode, limit to 1-2 suffixes
                terms_to_use = self.terms[:2] if len(self.terms) >= 2 else self.terms

            logger.info(
                "Using suffix-based adversarial queries",
                extra={
                    "intent": self.intent,
                    "suffix_count": len(terms_to_use),
                    "suffixes": terms_to_use[:3]
                }
            )

            for idx, term in enumerate(terms_to_use):
                cq = f"{query} {term}"
                try:
                    sub_kwargs = dict(kwargs)
                    sub_kwargs["skip_query_builder"] = True
                    docs, ids = await self.base(cq, **sub_kwargs)
                    if docs:
                        dialectical_docs.extend(docs)
                        dialectical_doc_ids.extend(ids)
                        logger.debug(f"Suffix query [{idx}] '{term}' returned {len(docs)} docs")
                except Exception as e:
                    logger.debug(f"Dialectical suffix query failed for term={term}: {e}")
                    continue

        self.last_metadata = {
            "base_docs": len(base_docs),
            "dialectical_docs": len(dialectical_docs),
            "dialectical_search_executed": True,
            "dialectical_zero_hit": len(dialectical_docs) == 0,
        }
        self.last_opposing_evidence_found = bool(dialectical_docs)

        # Score by inverse position; dialectical hits receive a multiplicative boost
        scored = {}
        for i, (d, doc_id) in enumerate(zip(base_docs, base_doc_ids)):
            pmid = str(doc_id) if doc_id is not None else f"b{i}"
            if pmid not in scored:
                scored[pmid] = {"doc": d, "doc_id": doc_id, "score": 1.0 / (i + 1)}

        for i, (d, doc_id) in enumerate(zip(dialectical_docs, dialectical_doc_ids)):
            pmid = str(doc_id) if doc_id is not None else f"d{i}"
            score = self.boost * (1.0 / (i + 1))
            if pmid in scored:
                scored[pmid]["score"] = max(scored[pmid]["score"], score)
            else:
                scored[pmid] = {"doc": d, "doc_id": doc_id, "score": score}

        # Sort by score and return top_k
        merged = sorted(scored.values(), key=lambda x: x["score"], reverse=True)
        merged_docs = [m["doc"] for m in merged][:top_k]
        merged_ids = [m.get("doc_id") for m in merged][:top_k]
        
        logger.info(
            "DialecticalRetriever merge complete",
            extra={
                "intent": self.intent,
                "base_docs": len(base_docs),
                "dialectical_docs": len(dialectical_docs),
                "unique_docs": len(scored),
                "returned": len(merged_docs)
            }
        )
        
        return merged_docs, merged_ids

    async def __call__(self, *args, intent: Optional[str] = None, **kwargs):
        """
        Async callable interface for dialectical retrieval.
        
        Args:
            *args: Positional arguments (typically query)
            intent: Clinical intent to guide adversarial query generation
            **kwargs: Additional keyword arguments
        """
        if intent:
            self.intent = intent
        return await self.retrieve(*args, intent=intent, **kwargs)
