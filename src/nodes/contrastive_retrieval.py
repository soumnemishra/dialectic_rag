import logging
import asyncio
import os
from typing import Dict, Any, List, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from src.core.registry import ModelRegistry, safe_ainvoke
from src.models.state import GraphState
from src.pubmed_client import PubMedClient
from src.retrieval.pico_extractor import PICOExtractor
from src.utils.debug_utils import get_debug_manager

logger = logging.getLogger(__name__)

QUERY_GENERATION_PROMPT = """
You are a clinical librarian. For each candidate diagnosis provided, generate two PubMed search queries based on the clinical vignette.
Each query must target a different evidence perspective.
The vignette:
{vignette}

The candidates:
{candidates}

For EACH candidate diagnosis, provide:
1. "supportive": A search query including the candidate name AND relevant positive findings/symptoms from the patient's presentation.
2. "challenging": A search query including the candidate name AND contradictory findings, alternative symptoms, or factors that argue against it.

IMPORTANT: Return ONLY a JSON object. The keys must be the exact candidate text provided, and the value must be an object with "supportive" and "challenging" keys containing the search strings.

Example format:
{{
  "Candidate A": {{
    "supportive": "\"Candidate A\" AND (symptom1 OR symptom2)",
    "challenging": "\"Candidate A\" AND (absent_symptom OR alternative_sign)"
  }}
}}
"""

class ContrastiveRetriever:
    def __init__(self, config_path: Optional[str] = None):
        self.llm = ModelRegistry.get_flash_llm(temperature=0.0, json_mode=True)
        self.prompt = ChatPromptTemplate.from_template(QUERY_GENERATION_PROMPT)
        self.pubmed = PubMedClient()
        self.config = self._load_config(config_path)

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        from pathlib import Path
        import yaml
        path = config_path or Path(__file__).resolve().parents[1] / "config" / "default.yaml"
        try:
            with open(path, "r") as f:
                return yaml.safe_load(f)
        except Exception:
            return {}

    async def generate_queries(self, vignette: str, candidates: List[str]) -> Dict[str, Dict[str, str]]:
        """Generate supportive and challenging queries per candidate using an LLM."""
        try:
            candidates_text = "\n".join([f"- {c}" for c in candidates])
            res = await safe_ainvoke(
                self.prompt | self.llm | JsonOutputParser(), 
                {"vignette": vignette, "candidates": candidates_text}
            )
            return res
        except Exception as e:
            logger.error(f"Query generation failed: {e}")
            return {}

async def contrastive_retrieval_node(state: GraphState) -> Dict[str, Any]:
    debug_manager = get_debug_manager()
    debug_retrieval = os.getenv("DEBUG_RETRIEVAL", "false").strip().lower() in {"1", "true", "yes", "on"}
    question = state.get("original_question", "")
    import re
    # Extract vignette
    parts = re.split(r"(?:\n\s*Options:\s*\n|\n\s*A:|\n\s*A\))", question, maxsplit=1)
    vignette = parts[0].strip()
    
    candidates = state.get("candidate_answers", [])
    if not candidates:
        logger.warning("No candidates found in state, using generic queries.")
        candidates = ["intervention"]
        
    retriever = ContrastiveRetriever()
    queries_by_candidate = await retriever.generate_queries(vignette, candidates)
    
    all_retrieved_docs: Dict[str, Any] = {}
    dedup_seen = set()
    
    # Store counts for trace
    results_summary = {}
    fallback_used = False
    fallback_queries = {}

    # Helper function to broaden a query
    def broaden_query(q: str) -> str:
        """Remove quotes, relax AND to OR, add fallback terms."""
        s = q.replace('"', '')
        s = s.replace(' AND ', ' OR ')
        suffix = ' OR ethics OR disclosure OR "medical error" OR "operative report" OR reporting'
        return s + suffix

    # First pass: try original queries
    for candidate, queries in queries_by_candidate.items():
        all_retrieved_docs[candidate] = []
        candidate_count = 0
        
        for p_type, qs in queries.items():
            try:
                # Sometimes LLM wraps queries in a list
                if isinstance(qs, list):
                    qs = qs[0]
                docs = await retriever.pubmed.search(qs, max_results=10)
                
                for doc in docs:
                    pmid = doc.pmid
                    if pmid not in dedup_seen:
                        dedup_seen.add(pmid)
                        # Tag document with candidate info and perspective
                        doc_dict = doc.model_dump()
                        doc_dict["candidate"] = candidate
                        doc_dict["perspective"] = p_type
                        all_retrieved_docs[candidate].append(doc_dict)
                        candidate_count += 1
                        
            except Exception as e:
                logger.error(f"Search failed for {candidate} ({p_type}): {e}")
                
        results_summary[candidate] = candidate_count

    # Second pass: if no results, try broadened fallback queries
    if len(dedup_seen) == 0:
        logger.info("No PMIDs found in initial queries. Running fallback broadened queries.")
        fallback_used = True
        for candidate, queries in queries_by_candidate.items():
            for p_type, qs in queries.items():
                try:
                    if isinstance(qs, list):
                        qs = qs[0]
                    alt_qs = broaden_query(qs)
                    fallback_queries[f"{candidate}_{p_type}"] = alt_qs
                    docs = await retriever.pubmed.search(alt_qs, max_results=20)
                    
                    for doc in docs:
                        pmid = doc.pmid
                        if pmid not in dedup_seen:
                            dedup_seen.add(pmid)
                            doc_dict = doc.model_dump()
                            doc_dict["candidate"] = candidate
                            doc_dict["perspective"] = p_type
                            doc_dict["query_fallback"] = True
                            all_retrieved_docs[candidate].append(doc_dict)
                            results_summary[candidate] = results_summary.get(candidate, 0) + 1
                except Exception as e:
                    logger.error(f"Fallback search failed for {candidate} ({p_type}): {e}")

    trace_event = {
        "node": "contrastive_retrieval",
        "section": "retrieval",
        "input": {"candidates": candidates},
        "evaluation_policy": {
            "zero_shot": True,
            "question_only_retrieval": True,
            "options_visible_to_retrieval": False,
            "mcq_options_present_but_hidden": bool(state.get("mcq_options")),
        },
        "output": {
            "queries_generated": queries_by_candidate,
            "results_per_candidate": results_summary,
            "total_unique": len(dedup_seen),
            "fallback_used": fallback_used,
            "fallback_queries": fallback_queries if fallback_used else None
        }
    }

    if debug_retrieval and debug_manager.is_enabled():
        retrieved_articles = []
        for _, articles in all_retrieved_docs.items():
            for article in articles:
                retrieved_articles.append(article)

        debug_manager.save_query_snapshot(
            query=question,
            payload={
                "pmids": sorted(list(dedup_seen)),
                "retrieved_articles": retrieved_articles,
                "queries_generated": queries_by_candidate,
                "results_per_candidate": results_summary,
                "fallback_used": fallback_used,
                "fallback_queries": fallback_queries if fallback_used else None,
                "evidence_scores": state.get("evidence_scores", {}),
                "calibration_metrics": state.get("calibration_metrics", {}),
                "final_answer": state.get("final_answer") or state.get("candidate_answer"),
                "epistemic_state": state.get("epistemic_state"),
                "trace_id": state.get("trace_id"),
            },
        )
        debug_manager.save_json(
            "workflow/state_snapshot.json",
            {
                "timestamp": state.get("trace_created_at"),
                "node": "contrastive_retrieval",
                "input_state": dict(state),
                "output_state": {
                    "retrieved_docs": all_retrieved_docs,
                    "trace_event": trace_event,
                },
            },
        )
    
    return {
        "retrieved_docs": all_retrieved_docs,
        "trace_events": [trace_event]
    }
