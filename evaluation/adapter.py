from dataclasses import dataclass, field
import uuid
from datetime import datetime
from typing import List, Dict, Any
import logging

from src.orchestrator.graph import build_graph
from src.state.state import GraphState
from src.config import is_evaluation_mode

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
        # Initialize registry before building graph
        from src.agents.registry import AgentRegistry
        AgentRegistry.get_instance().initialize()
        
        self.graph = build_graph()
        
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
            "chat_history": [],
            "trace_id": str(uuid.uuid4()),
            "trace_created_at": datetime.utcnow().isoformat(),
            "trace_events": [],
            "answer": "UNKNOWN",
            "predicted_letter": "UNKNOWN",
            "answer_source": "rag_direct",
            "plan": [],
            "plan_complexity": "simple",
            "plan_error": None,
            "past_exp": [],
            "final_answer": "",
            "final_answer_raw": "",
            "final_answer_aligned": "",
            "step_output": [],
            "step_docs_ids": [],
            "step_notes": [],
            "candidate_answer": "UNKNOWN",
            "candidate_answer_prev": "",
            "candidate_switch_reason": "",
            "intent": "informational",
            "risk_level": "low",
            "confidence": 0.0,
            "reasoning": "",
            "safety_flags": [],
            "needs_guidelines": False,
            "requires_disclaimer": False,
            "router_output": {
                "execution_mode": "multihop",
                "requires_planning": True,
                "requires_extraction": True,
                "requires_evidence_grading": True,
                "answer_policy": {},
                "execution_budget": None,
            },
            "evidence_polarity": {
                "polarity": "insufficient",
                "confidence": 0.0,
                "reasoning": "",
            },
            "evidence_decision": "accept",
            "retry_count": 0,
            "current_documents": [],
            "current_doc_ids": [],
            "tcs_score": 0.0,
            "temporal_conflicts": [],
            "rps_scores": [],
            "thesis_docs": [],
            "antithesis_docs": [],
            "dialectic_synthesis": {},
            "controversy_label": "UNKNOWN",
            "eus_per_claim": {},
            "belief_intervals": {},
            "eus_override": None,
            "evaluation_mode": is_evaluation_mode(),
        }
        
        try:
            # Run the graph
            result = await self.graph.ainvoke(initial_state)

            rps_scores = result.get("rps_scores", [])
            rps_avg = sum(s.get("rps_score", 0.0) for s in rps_scores) / max(len(rps_scores), 1)
            safety_flags = result.get("safety_flags", [])
            safety_intercepted = any(
                flag not in {"eval_mode_safety_skip", "skipped_empty", "audit_error", "node_crash"}
                for flag in safety_flags
            )
            # Build metadata; answer_source may be overridden below with heuristics
            metadata = {
                "tcs_score": result.get("tcs_score", 0.0),
                "rps_avg": rps_avg,
                "applicability_score": result.get("applicability_score", 0.5),
                "controversy_label": result.get("controversy_label", "UNKNOWN"),
                "answer_source": result.get("answer_source", None),
                "eus": result.get("eus_per_claim", {}).get("global"),
                "belief_intervals": result.get("belief_intervals", {}),
                "trace_id": result.get("trace_id"),
                "trace_event_count": len(result.get("trace_events", []) or []),
                "trace_events": result.get("trace_events", []),
                "dialectic_gate_triggered": False,
                "dialectic_gate_reason": None,
                "eus_belief": {
                    "belief": result.get("belief_intervals", {}).get("global", {}).get("belief"),
                    "plausibility": result.get("belief_intervals", {}).get("global", {}).get("plausibility"),
                    "uncertainty": result.get("eus_per_claim", {}).get("global"),
                },
                "risk_level": result.get("risk_level", "unknown"),
                "safety_flags": safety_flags,
                "safety_intercepted": safety_intercepted,
            }
            
            final_answer = result.get("final_answer", "")
            predicted_letter = str(result.get("predicted_letter", "UNKNOWN")).upper()
            if predicted_letter not in {"A", "B", "C", "D", "UNKNOWN"}:
                predicted_letter = "UNKNOWN"
            if "**Final Answer:" not in final_answer:
                plan_summary = result.get("plan_summary", {})
                if not plan_summary:
                    past_exp = result.get("past_exp", [])
                    if past_exp:
                        plan_summary = past_exp[-1].get("plan_summary", {})
                decision = plan_summary.get("final_decision") if isinstance(plan_summary, dict) else None
                if not decision:
                    decision = predicted_letter
                if decision:
                    final_answer += f"\n\n**Final Answer: {decision}**"
            if not final_answer:
                # Fallback if final_answer not set (e.g. error)
                past_exp = result.get("past_exp", [])
                if past_exp:
                    last_exp = past_exp[-1]
                    summary = last_exp.get("plan_summary", {})
                    final_answer = summary.get("answer", "No answer generated.")
                else:
                    final_answer = "No answer generated."

            print(f"\n[DEBUG] Raw Model Output: {final_answer[:500]}...\n")
            
            # Collect sources (PMIDs or URLs from document objects)
            # MA-RAG stores doc IDs in step_docs_ids
            all_sources = set()
            past_exp = result.get("past_exp", [])
            for exp in past_exp:
                step_ids = exp.get("step_docs_ids", [])
                for ids_list in step_ids:
                    if isinstance(ids_list, list):
                        for doc_id in ids_list:
                            all_sources.add(doc_id)
                    else:
                        all_sources.add(ids_list)

            # Collect per-step contradiction diagnostics if available
            contrad_maps = []
            contrad_scores = []
            for out in result.get("step_output", []) or []:
                if isinstance(out, dict):
                    if out.get("contradiction_map"):
                        contrad_maps.append(out.get("contradiction_map"))
                    if out.get("contradiction_score") is not None:
                        try:
                            contrad_scores.append(float(out.get("contradiction_score") or 0.0))
                        except Exception:
                            pass
            metadata["contradiction_maps"] = contrad_maps
            metadata["contradiction_scores"] = contrad_scores
            metadata["contradiction_score_avg"] = sum(contrad_scores) / len(contrad_scores) if contrad_scores else 0.0

            for event in result.get("trace_events", []) or []:
                if not isinstance(event, dict):
                    continue
                if str(event.get("section", "")).lower() != "dialectic_gate":
                    continue
                data = event.get("data", {}) if isinstance(event.get("data", {}), dict) else {}
                if str(data.get("decision", "")).strip() == "adversarial_retrieval":
                    metadata["dialectic_gate_triggered"] = True
                    metadata["dialectic_gate_reason"] = data.get("reason")
                    break

            # Derive answer_source robustly when the graph didn't explicitly set it.
            derived = str(result.get("answer_source") or "").strip() or None
            if derived:
                metadata["answer_source"] = derived
            else:
                # If plan present with executed steps, call it plan_execute
                has_plan = bool(result.get("plan")) and len(result.get("plan")) > 0
                has_step_output = bool(result.get("step_output")) and len(result.get("step_output")) > 0
                # past_exp sometimes contains per-run step outputs
                past_exp = result.get("past_exp", []) or []
                past_has_steps = any((p.get("step_output") or p.get("step_docs_ids")) for p in past_exp)

                if has_plan and (has_step_output or past_has_steps):
                    metadata["answer_source"] = "plan_execute"
                elif result.get("thesis_docs") or result.get("antithesis_docs"):
                    # dialectical retrieval path
                    metadata["answer_source"] = "dialectical"
                elif result.get("supplemental_retrieval_triggered"):
                    metadata["answer_source"] = "supplemental_retrieval"
                else:
                    metadata["answer_source"] = "rag_direct"
            
            # --- Observability: Assemble and Report Trace ---
            try:
                from src.utils.trace_reporter import TraceReporter
                # Inject question_id into state for the reporter
                result["question_id"] = question_id or metadata.get("question_id", "eval_query")
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
