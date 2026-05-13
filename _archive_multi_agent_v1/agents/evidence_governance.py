import logging
from typing import Dict, Any, List, Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential_jitter, RetryError

from src.state.state import GraphState
from src.core.registry import ModelRegistry, safe_ainvoke
from src.prompts.templates import with_json_system_suffix
from src.utils.epistemic_trace import build_trace_event, build_trace_updates

logger = logging.getLogger(__name__)

class EvidenceGovernanceOutput(BaseModel):
    """
    Pydantic schema for the epistemic governance decision.
    """
    decision: Literal["accept", "abstain"] = Field(
        description="accept | abstain"
    )
    reasoning: str = Field(
        description="Brief reasoning for the decision"
    )

EVIDENCE_GOVERNANCE_SYSTEM_PROMPT = """You are the Epistemic Governance Module for a clinical RAG system.
Your goal is to decide whether the current retrieved evidence is scientifically robust enough to support a clinical recommendation, or if the system must abstain.

STRICT GOVERNANCE RULES:
1. ABSTAIN if:
   - Polarity is 'insufficient'.
   - Average RPS is near zero (< 0.1) and evidence is mixed.
   - Clinical risk is 'high' and evidence confidence is low (< 0.6).
2. ACCEPT if:
   - Evidence provides clear support or refutation with reasonable confidence.
   - Even if evidence is 'weak_support', if RPS is high (> 0.5), we may accept but note the uncertainty.

OUTPUT FORMAT (JSON):
{{
    "decision": "accept" | "abstain",
    "reasoning": "One sentence epistemic justification."
}}
"""

EVIDENCE_GOVERNANCE_HUMAN_PROMPT = """Question: {question}
Polarity: {polarity} (Confidence: {confidence})
Average RPS: {avg_rps:.2f}
TCS (Temporal Shift): {tcs_score:.2f}
Applicability: {applicability_score:.2f}
Risk Level: {risk_level}"""

class EvidenceGovernanceModule:
    """
    Quality gate that decides whether the retrieved evidence is sufficient to proceed.
    Focuses on 'accept' vs 'abstain' to ensure clinical safety.
    """

    def __init__(self):
        self.llm = ModelRegistry.get_flash_llm(temperature=0.0, json_mode=True)
        if self.llm is None:
            raise RuntimeError("EvidenceGovernanceModule: Flash LLM failed to load.")

        self.parser = JsonOutputParser(pydantic_object=EvidenceGovernanceOutput)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", with_json_system_suffix(EVIDENCE_GOVERNANCE_SYSTEM_PROMPT)),
            ("human",  EVIDENCE_GOVERNANCE_HUMAN_PROMPT),
        ])
        self.chain = self.prompt | self.llm | self.parser

    @retry(stop=stop_after_attempt(2), wait=wait_exponential_jitter(initial=0.5, max=6))
    async def _invoke_chain(self, inputs: dict) -> dict:
        return await safe_ainvoke(self.chain, inputs)

    async def govern(self, state: GraphState) -> Dict[str, Any]:
        try:
            # --- EPISTEMIC STATE EXTRACTION ---
            evidence_polarity = state.get("evidence_polarity", {})
            polarity   = evidence_polarity.get("polarity",   "insufficient")
            confidence = float(evidence_polarity.get("confidence", 0.0))
            
            tcs_score = float(state.get("tcs_score", 0.0))
            applicability_score = float(state.get("applicability_score", 0.0))
            risk_level = state.get("risk_level", "low")
            
            rps_scores = state.get("rps_scores", [])
            valid_rps = [float(s.get("final_score", s.get("rps_score", 0.5))) for s in rps_scores if s.get("final_score") or s.get("rps_score")]
            avg_rps = sum(valid_rps) / len(valid_rps) if valid_rps else 0.5

            inputs = {
                "question": state["original_question"],
                "polarity": polarity,
                "confidence": round(confidence, 3),
                "avg_rps": round(avg_rps, 3),
                "tcs_score": round(tcs_score, 3),
                "applicability_score": round(applicability_score, 3),
                "risk_level": risk_level,
            }

            # Hard stop for extremely low quality
            if polarity == "insufficient" and avg_rps < 0.1:
                return self._build_result("abstain", "Extremely low quality evidence with insufficient polarity.", inputs, state)

            try:
                result = await self._invoke_chain(inputs)
            except Exception as e:
                logger.error(f"Governance chain failed: {e}")
                return {"governance_decision": "accept"}

            decision = result.get("decision", "accept")
            reason = result.get("reasoning", "")
            
            return self._build_result(decision, reason, inputs, state)

        except Exception as e:
            logger.error(f"Governance failed: {e}", exc_info=True)
            return {"governance_decision": "accept"}

    def _build_result(self, decision: str, reason: str, inputs: dict, state: GraphState) -> dict:
        trace_event = build_trace_event(
            state,
            section="decision_governance",
            event="governance_decision",
            node="evidence_governance",
            data={"inputs": inputs, "decision": decision, "reason": reason},
            influence={"routing": decision},
            attach_context=False,
        )
        trace_updates = build_trace_updates(state, [trace_event])
        return {
            "governance_decision": decision,
            "trace_events": trace_updates.get("trace_events", []),
            "trace_id": trace_updates.get("trace_id"),
            "trace_created_at": trace_updates.get("trace_created_at"),
        }

async def evidence_governance_node(state: GraphState) -> Dict[str, Any]:
    from src.agents.registry import AgentRegistry
    # Note: We'll need to update AgentRegistry to include evidence_governance
    try:
        # For now, we'll instantiate directly or use a temporary hack
        # In a real refactor, AgentRegistry would be updated.
        module = EvidenceGovernanceModule()
        return await module.govern(state)
    except Exception as e:
        logger.error(f"evidence_governance_node crashed: {e}", exc_info=True)
        return {"governance_decision": "accept"}
