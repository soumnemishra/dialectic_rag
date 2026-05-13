from dataclasses import dataclass, field
import uuid
from datetime import datetime
from typing import List, Dict, Any
import logging

from src.graph.workflow import build_workflow
from src.models.state import GraphState
from src.config import settings

logger = logging.getLogger(__name__)

@dataclass
class AgentResult:
    answer: str
    sources: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

class MaRagAdapter:
    """
    Adapter to make MA-RAG pipeline compatible with PubMedQAEvaluator.
    """
    
    def __init__(self):
        self.graph = build_workflow()
        
    async def answer_query(self, prompt: str, question_id: str | None = None) -> AgentResult:

        """
        Run the pipeline and return result in expected format.
        """
        # Prompt formatting is handled upstream (run_evaluation.py). This adapter
        # passes the question through without adding or modifying options.
        
        # QOR (Question-Only Retrieval) Parsing: Separate clinical vignette from options
        import re
        clinical_vignette = prompt
        mcq_options = ""
        
        # Match standard dataset delimiters for MCQ options
        match = re.split(r"(?:\n\s*Options:\s*\n|\n\s*A:|\n\s*A\))", prompt, maxsplit=1)
        if len(match) == 2:
            clinical_vignette = match[0].strip()
            # Restore the prefix if it was split by the option letter itself
            split_str = prompt[len(match[0]):][:len(prompt)-len(match[0])-len(match[1])]
            mcq_options = (split_str + match[1]).strip()
            logger.info("QOR parsing successful. Separated vignette and options.")
            
        initial_state: GraphState = {
            "original_question": f"{clinical_vignette}\n\nOptions:\n{mcq_options}" if mcq_options else clinical_vignette,
            "mcq_options": mcq_options,
            "trace_id": str(uuid.uuid4()),
            "trace_created_at": datetime.utcnow().isoformat(),
            "trace_events": [],
            "evidence_pool": [],
            "retrieved_docs": {},
            "step_notes": [],
            "candidate_answer": "UNKNOWN",
            "final_reasoning": "",
            "safety_flags": [],
        }
        
        try:
            # Run the graph
            result = await self.graph.ainvoke(initial_state)
            ep_result = result.get("epistemic_result")
            
            # Extract metrics from EpistemicResult
            belief = ep_result.belief if ep_result else 0.5
            uncertainty = ep_result.uncertainty if ep_result else 0.5
            conflict = ep_result.conflict if ep_result else 0.0
            
            # RPS and Applicability from pool
            pool = result.get("evidence_pool", [])
            rps_avg = sum(i.reproducibility_score for i in pool) / max(len(pool), 1)
            applic_avg = sum(i.applicability_score for i in pool) / max(len(pool), 1)

            safety_flags = result.get("safety_flags", [])
            
            metadata = {
                "tcs_score": conflict, # Map conflict to tcs for eval compatibility
                "rps_avg": rps_avg,
                "applicability_score": applic_avg,
                "controversy_label": str(ep_result.state if ep_result else "UNKNOWN"),
                "answer_source": "dialectic_rag",
                "eus": uncertainty,
                "belief_intervals": {"global": {"belief": belief}},
                "trace_id": result.get("trace_id"),
                "trace_events": result.get("trace_events", []),
                "eus_belief": {
                    "belief": belief,
                    "uncertainty": uncertainty,
                },
                "risk_level": result.get("risk_level", "unknown"),
                "safety_flags": safety_flags,
            }


            # Final Answer Extraction
            final_answer = result.get("candidate_answer", "")
            
            # Heuristic to find the bracketed letter [A/B/C/D]
            import re
            letter_match = re.search(r"\*\*Final Answer:\s*\[?([A-D])\]?\*\*", final_answer)
            predicted_letter = letter_match.group(1) if letter_match else "UNKNOWN"
            
            # Sources from evidence_pool
            pool = result.get("evidence_pool", [])
            all_sources = [item.pmid for item in pool]

            
            # --- Observability: Assemble and Report Trace ---
            try:
                from src.utils.trace_reporter import TraceReporter
                # Inject question_id and predicted_letter into state for the reporter
                result["question_id"] = question_id or metadata.get("question_id", "eval_query")
                result["final_answer"] = final_answer
                result["predicted_letter"] = predicted_letter
                structured_trace = TraceReporter.assemble(result)

                TraceReporter.save_trace(structured_trace)
                TraceReporter.print_summary(structured_trace)
                
                # Update metadata with causal summary
                metadata["causal_analysis"] = structured_trace.get("causal_analysis")
            except Exception as tr_exc:
                logger.warning("TraceReporter failed: %s", tr_exc)

            return AgentResult(
                answer=final_answer,
                sources=list(all_sources),
                metadata=metadata,
            )

            
        except Exception as e:
            logger.error(f"Adapter failed: {e}")
            return AgentResult(answer=f"Error: {e}", sources=[], metadata={})
