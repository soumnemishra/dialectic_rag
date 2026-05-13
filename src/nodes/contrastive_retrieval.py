import logging
import asyncio
from typing import Dict, Any, List, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from src.core.registry import ModelRegistry, safe_ainvoke
from src.models.state import GraphState
from src.pubmed_client import PubMedClient
from src.retrieval.pico_extractor import PICOExtractor

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

    trace_event = {
        "node": "contrastive_retrieval",
        "section": "retrieval",
        "input": {"candidates": candidates},
        "output": {"queries_generated": queries_by_candidate, "results_per_candidate": results_summary, "total_unique": len(dedup_seen)}
    }
    
    return {
        "retrieved_docs": all_retrieved_docs,
        "trace_events": [trace_event]
    }
