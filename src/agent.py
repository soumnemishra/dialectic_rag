# FILE: src/agent.py
"""
MedicalAgent - High-level interface for the MA-RAG pipeline.

This module provides a simple interface for the Streamlit app to interact
with the multi-agent RAG system.

Example:
    agent = MedicalAgent()
    response, query_log, reasoning = await agent.chat("What is the treatment for diabetes?")
"""

import logging
import uuid
from datetime import datetime
from typing import List, Dict, Any, Tuple

from src.graph.workflow import build_workflow
from src.models.state import GraphState
from src.config import is_evaluation_mode

logger = logging.getLogger(__name__)


class MedicalAgent:
    """
    High-level interface for medical question answering using MA-RAG.
    
    Wraps the LangGraph orchestration pipeline and provides a simple
    async interface for chat-based interactions.
    """
    
    def __init__(self):
        """Initialize the MedicalAgent with the MA-RAG graph."""
        logger.info("Initializing MedicalAgent...")
        self.graph = build_workflow()
    
    async def chat(
        self, 
        query: str, 
        history: List[Dict[str, str]] = None
    ) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
        """
        Process a medical question and return the answer.
        
        Returns:
            Tuple of (answer, query_log, reasoning_steps, risk_metadata)
        """
        logger.info("Received query: %s", query)
        try:
            logger.info("Invoking graph with query...")
            
            # Initialize the graph state
            initial_state: GraphState = {
                "original_question": query,
                "chat_history": history or [],
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
                "dialectical_retrieval_done": False,
                "controversy_label": "UNKNOWN",
                "eus_per_claim": {},
                "belief_intervals": {},
                "eus_override": None,
                "evaluation_mode": is_evaluation_mode(),
            }
            
            # Run the graph asynchronously
            result = await self.graph.ainvoke(initial_state)
            
            # Extract the final answer
            final_answer = result.get("final_answer", "I was unable to generate an answer.")
            predicted_letter = str(result.get("predicted_letter", "UNKNOWN")).upper()
            if predicted_letter not in {"A", "B", "C", "D", "UNKNOWN"}:
                predicted_letter = "UNKNOWN"
            if final_answer and "**Final Answer:" not in final_answer:
                final_answer += f"\n\n**Final Answer: {predicted_letter}**"
            
            # Build query log from past experiences
            query_log = []
            reasoning_steps = []
            
            # Extract reasoning from past experiences
            past_exp = result.get("past_exp", [])
            if past_exp:
                exp = past_exp[0]  # Get the first (and usually only) execution
                
                # Add planning phase
                plan = exp.get("plan", [])
                if plan:
                    plan_preview = [
                        (p.get("question", str(p)) if isinstance(p, dict) else str(p))
                        for p in plan[:3]
                    ]
                    reasoning_steps.append({
                        "phase": "PLANNING",
                        "thought": f"Breaking down the question into {len(plan)} steps",
                        "details": ", ".join(plan_preview) + ("..." if len(plan) > 3 else "")
                    })
                
                # Add execution phases
                step_questions = exp.get("step_question", [])
                step_outputs = exp.get("step_output", [])
                
                for i, (sq, so) in enumerate(zip(step_questions, step_outputs)):
                    task_type = sq.get("type", "question-answering")
                    task = sq.get("task", "")
                    answer = so.get("answer", "")[:200]
                    
                    phase = "SEARCHING" if task_type == "question-answering" else "SYNTHESIZING"
                    reasoning_steps.append({
                        "phase": phase,
                        "thought": task[:100],
                        "details": answer[:150] + ("..." if len(answer) > 150 else "")
                    })
                
                # Add final synthesis phase
                reasoning_steps.append({
                    "phase": "SYNTHESIZING",
                    "thought": "Combining all findings into final answer",
                    "details": ""
                })
            
            logger.info(f"Query processed successfully. Answer length: {len(final_answer)}")
            
            # Extract risk metadata for UI display
            rps_scores = result.get("rps_scores", [])
            risk_metadata = {
                "risk_level": result.get("risk_level", "low"),
                "intent": result.get("intent", "informational"),
                "requires_disclaimer": result.get("requires_disclaimer", False),
                "needs_guidelines": result.get("needs_guidelines", False),
                "tcs_score": result.get("tcs_score", 0.0),
                "answer_source": result.get("answer_source", "rag_direct"),
                "eus": result.get("eus_per_claim", {}).get("global"),
                "applicability_score": result.get("applicability_score"),
                "controversy_label": result.get("controversy_label", "UNKNOWN"),
                "temporal_conflicts": result.get("temporal_conflicts", []),
                "eus_summary": result.get("belief_intervals", {}).get("global", {}),
                "rps_avg": (
                    sum(score.get("final_score", 0.0) for score in rps_scores)
                    / max(len(rps_scores), 1)
                ),
            }
            
            return final_answer, query_log, reasoning_steps, risk_metadata
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            error_response = f"I encountered an error while processing your question: {str(e)}"
            return error_response, [], [], {"risk_level": "unknown", "intent": "error"}
    
    async def answer_query(self, query: str) -> "AgentResult":
        """
        Simplified interface for evaluation.
        """
        response, _, _, _ = await self.chat(query)
        return AgentResult(answer=response, sources=[])


class AgentResult:
    """Result from agent query for evaluation compatibility."""
    
    def __init__(self, answer: str, sources: List[str]):
        self.answer = answer
        self.sources = sources
