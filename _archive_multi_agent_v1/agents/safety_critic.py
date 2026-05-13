import logging
import re
from typing import Dict, Any, List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential_jitter, RetryError

from src.state.state import GraphState
from src.query_builder import parse_markdown_json
from src.prompts.templates import SAFETY_CRITIC_SYSTEM_PROMPT, SAFETY_CRITIC_HUMAN_PROMPT, with_json_system_suffix
from src.core.registry import ModelRegistry, safe_ainvoke
from src.utils.epistemic_config import get_epistemic_setting
from src.utils.epistemic_trace import build_trace_event, build_trace_updates

logger = logging.getLogger(__name__)


DEFAULT_DISCLAIMER = (
    "This response is for informational purposes only and is not a substitute for "
    "professional medical advice or in-person evaluation."
)

TRANSPARENCY_DISCLAIMER = (
    "Note: Retrieved literature did not contain sufficient evidence to definitively answer this query. "
    "This answer was synthesized using general medical knowledge."
)


def _has_disclaimer(answer: str) -> bool:
    lowered = (answer or "").lower()
    return any(
        phrase in lowered
        for phrase in (
            "not a substitute for professional medical advice",
            "for informational purposes only",
            "consult a healthcare professional",
            "seek medical attention",
            "in-person evaluation",
        )
    )


def _ensure_disclaimer(answer: str) -> str:
    if answer.startswith(DEFAULT_DISCLAIMER):
        return answer
    if _has_disclaimer(answer):
        return answer
    return f"{DEFAULT_DISCLAIMER}\n\n{answer}".strip()


def _is_citation_related_issue(issue: str) -> bool:
    lowered = (issue or "").lower()
    return any(
        marker in lowered
        for marker in (
            "pmid",
            "citation",
            "reference",
            "references",
            "hallucination",
            "unsupported by cited",
            "missing cited",
        )
    )


def _should_exempt_citations(answer: str, answer_source: str) -> bool:
    lowered_answer = (answer or "").lower()
    source = (answer_source or "").strip().lower()
    return source == "general_knowledge" or TRANSPARENCY_DISCLAIMER.lower() in lowered_answer


def _extract_final_answer_letter(answer: str, fallback: str = "UNKNOWN") -> str:
    """Recover the chosen option letter from answer text or fall back to state."""
    if isinstance(answer, str):
        patterns = (
            r"(?is)\*\*final\s+answer:\s*\[?([A-D]|UNKNOWN)\]?\*\*",
            r"(?is)final\s+answer:\s*\*?\*?\[?([A-D]|UNKNOWN)\]?\*?\*?",
            r"(?is)\\boxed\{([A-D]|UNKNOWN)\}",
        )
        for pattern in patterns:
            match = re.search(pattern, answer)
            if match:
                letter = match.group(1).upper()
                if letter in {"A", "B", "C", "D"}:
                    return letter

        stripped = answer.strip().upper()
        if stripped in {"A", "B", "C", "D"}:
            return stripped

    fallback = str(fallback or "UNKNOWN").strip().upper()
    if fallback in {"A", "B", "C", "D"}:
        return fallback
    return "UNKNOWN"


def _ensure_final_answer_tag(answer: str, letter: str) -> str:
    """Remove any existing final-answer tag and append the canonical one."""
    cleaned = re.sub(r"(?is)\*\*final\s+answer:\s*(?:[A-D]|UNKNOWN)\*\*", "", answer or "").strip()
    tag = f"**Final Answer: {letter}**"
    if cleaned:
        return f"{cleaned}\n\n{tag}"
    return tag


# ------------------------------------------------------------------ #
#  Output schema                                                      #
# ------------------------------------------------------------------ #

class SafetyCriticOutput(BaseModel):
    """
    Pydantic schema for safety audit output.

    Why: bare JsonOutputParser() silently returns is_safe=True when
    the LLM uses 'safe' instead of 'is_safe'. Every unsafe answer
    would then pass the audit with no warning.
    """
    is_safe:        bool            = Field(default=True)
    issues:         List[str]       = Field(default_factory=list)
    refined_answer: str | None      = Field(default=None)


# ------------------------------------------------------------------ #
#  ClinicalSafetyCriticAgent                                          #
# ------------------------------------------------------------------ #

class ClinicalSafetyCriticAgent:
    """
    Audits the final answer for clinical safety compliance before
    it is returned to the user.

    Checklist (enforced via SAFETY_CRITIC_SYSTEM_PROMPT):
      1. No inappropriate absolutes ("always", "never", "cure")
      2. Appropriate medical uncertainty language
      3. Disclaimer present for high/medium risk queries
      4. Claims backed by cited PMIDs
      5. Drug contraindications mentioned where relevant
      6. PMID citations preserved in refined answer
      7. Final Answer tag preserved if present
      8. Conflicting evidence acknowledged when polarity is refute/mixed

    Evaluation mode:
      When evaluation_mode=True, unsafe answers are NOT refined — the
      original answer is preserved for fair benchmark scoring. Issues
      are still recorded in safety_flags and an "eval_mode_safety_skip"
      flag is added so metrics can separate "genuinely safe" from
      "skipped for eval".
    """

    def __init__(self):
        use_flash = bool(get_epistemic_setting(
            "safety.use_flash_llm",
            False,
            env_var="MRAGE_SAFETY_USE_FLASH",
        ))

        self.llm = None
        if use_flash:
            self.llm = ModelRegistry.get_flash_llm(temperature=0.0, json_mode=True)
            if self.llm is None:
                logger.info("Safety critic flash LLM unavailable; falling back to heavy LLM.")

        if self.llm is None:
            self.llm = ModelRegistry.get_heavy_llm(temperature=0.0, json_mode=True)

        if self.llm is None:
            raise RuntimeError(
                "ClinicalSafetyCriticAgent: Heavy LLM failed to load. "
                "Check Ollama is running or GOOGLE_API_KEY is set."
            )

        self.parser = JsonOutputParser(pydantic_object=SafetyCriticOutput)

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", with_json_system_suffix(SAFETY_CRITIC_SYSTEM_PROMPT)),
            ("human",  SAFETY_CRITIC_HUMAN_PROMPT),
        ])

        # Chain built ONCE in __init__
        self.chain = self.prompt | self.llm | self.parser
        self.raw_chain = self.prompt | self.llm

    # ---------------------------------------------------------------- #
    #  Private helpers                                                   #
    # ---------------------------------------------------------------- #

    @retry(stop=stop_after_attempt(2), wait=wait_exponential_jitter(initial=0.5, max=6))
    async def _invoke_chain(self, inputs: dict) -> dict:
        """Runs audit chain with up to 2 retries on transient failures."""
        try:
            return await safe_ainvoke(self.chain, inputs)
        except RetryError:
            raw_result = await safe_ainvoke(self.raw_chain, inputs)
            raw_text = raw_result.content if hasattr(raw_result, "content") else str(raw_result)
            return parse_markdown_json(raw_text)

    def _extract_polarity_string(self, state: GraphState) -> str:
        """
        Safely extract polarity string for prompt injection.
        SAFETY_CRITIC_HUMAN_PROMPT has {evidence_polarity} placeholder —
        this must always be a non-None string or LangChain raises KeyError.
        """
        polarity_data = state.get("evidence_polarity", {})
        if isinstance(polarity_data, dict):
            return polarity_data.get("polarity", "insufficient")
        if isinstance(polarity_data, str):
            return polarity_data
        return "insufficient"

    # ---------------------------------------------------------------- #
    #  Public interface                                                  #
    # ---------------------------------------------------------------- #

    async def critique(self, state: GraphState) -> Dict[str, Any]:
        """
        Audit the final answer and either pass it, refine it, or flag it.

        Reads from GraphState:
            final_answer      — the answer to audit
            intent            — from ClinicalIntentAgent
            risk_level        — from ClinicalIntentAgent
            evidence_polarity — from EvidencePolarityAgent
            evaluation_mode   — if True, preserve original for benchmarking
            safety_flags      — existing flags (appended to, not overwritten)

        Returns GraphState update:
            final_answer      — original, refined, or warning-appended
            safety_flags      — list of issues found (empty = safe)
            predicted_letter  — preserved MedQA option letter
        """
        try:
            answer         = state.get("final_answer", "")
            intent         = state.get("intent",     "unknown")
            risk           = state.get("risk_level", "low")
            eval_mode      = state.get("evaluation_mode", False)
            answer_source  = state.get("answer_source", "rag_direct")
            existing_flags = state.get("safety_flags", [])
            predicted_letter = _extract_final_answer_letter(answer, state.get("predicted_letter", "UNKNOWN"))
            polarity_str   = self._extract_polarity_string(state)

            # Skip audit for empty or already-errored answers
            if not answer or "Error" in answer:
                logger.info("Safety audit skipped — empty or error answer.")
                return {
                    "safety_flags": existing_flags + ["skipped_empty"],
                    "predicted_letter": predicted_letter,
                }

            logger.info(f"Safety audit | risk={risk} intent={intent} polarity={polarity_str}")
            risk_level = str(risk).lower()
            needs_disclaimer = risk_level in {"high", "medium"} or bool(state.get("requires_disclaimer", False))

            # IMPORTANT: The upstream generator is instructed to omit disclaimers.
            # To avoid false audit failures (and unnecessary refinements) we
            # add the disclaimer *before* auditing when it is required.
            answer_for_audit = _ensure_disclaimer(answer) if needs_disclaimer else answer

            try:
                result = await self._invoke_chain({
                    "answer":            answer_for_audit,
                    "intent":            intent,
                    "risk_level":        risk,
                    "evidence_polarity": polarity_str,   # required by updated template
                    "answer_source":     answer_source,
                })
            except RetryError as e:
                raw_result = await safe_ainvoke(self.raw_chain, {
                    "answer":            answer_for_audit,
                    "intent":            intent,
                    "risk_level":        risk,
                    "evidence_polarity": polarity_str,
                    "answer_source":     answer_source,
                })
                raw_text = raw_result.content if hasattr(raw_result, "content") else str(raw_result)
                result = parse_markdown_json(raw_text)

            is_safe  = result.get("is_safe", True)
            issues   = result.get("issues", [])
            refined  = result.get("refined_answer")

            if _should_exempt_citations(answer, answer_source):
                filtered_issues = [issue for issue in issues if not _is_citation_related_issue(issue)]
                if len(filtered_issues) != len(issues):
                    logger.info(
                        "Safety audit: citation exemption applied | source=%s",
                        answer_source,
                    )
                issues = filtered_issues
                if not issues:
                    is_safe = True

            if not is_safe:
                logger.warning(f"Safety FAILED | issues: {issues}")

                # EVALUATION MODE — preserve original answer for fair benchmarking
                # but record that safety was skipped so metrics are honest
                if eval_mode:
                    logger.info("Eval mode: preserving original answer despite safety issues.")
                    return {
                        "final_answer": _ensure_final_answer_tag(
                            _ensure_disclaimer(answer) if needs_disclaimer else answer,
                            predicted_letter,
                        ),
                        "safety_flags": existing_flags + issues + ["eval_mode_safety_skip"],
                        "predicted_letter": predicted_letter,
                    }

                # STANDARD MODE — apply refinement or append warning
                if refined:
                    logger.info("Applying refined safe answer.")
                    refined = _ensure_disclaimer(refined) if needs_disclaimer else refined
                    refined_letter = _extract_final_answer_letter(refined, predicted_letter)
                    return {
                        "final_answer": _ensure_final_answer_tag(refined, refined_letter),
                        "safety_flags": existing_flags + issues,
                        "predicted_letter": refined_letter,
                    }
                else:
                    safe_answer = _ensure_disclaimer(answer) if needs_disclaimer else answer
                    warning = (
                        "\n\n**SAFETY WARNING**: This content has been flagged for review: "
                        + "; ".join(issues)
                    )
                    return {
                        "final_answer": _ensure_final_answer_tag(safe_answer + warning, predicted_letter),
                        "safety_flags": existing_flags + issues,
                        "predicted_letter": predicted_letter,
                    }

            logger.info("Safety audit PASSED.")
            final_answer = _ensure_disclaimer(answer) if needs_disclaimer else answer
            if final_answer != answer:
                return {
                    "final_answer": _ensure_final_answer_tag(final_answer, predicted_letter),
                    "safety_flags": existing_flags + ["disclaimer_added"],
                    "predicted_letter": predicted_letter,
                }
            return {
                "safety_flags": existing_flags,
                "predicted_letter": predicted_letter,
            }

        except Exception as e:
            logger.error(f"Safety audit failed: {e}", exc_info=True)
            return {
                "safety_flags": state.get("safety_flags", []) + ["audit_error"],
                "predicted_letter": state.get("predicted_letter", "UNKNOWN"),
            }


# ------------------------------------------------------------------ #
#  LangGraph node wrapper                                             #
# ------------------------------------------------------------------ #

from src.agents.registry import AgentRegistry


async def safety_critic_node(state: GraphState) -> Dict[str, Any]:
    """Uses registry singleton."""
    try:
        agent = AgentRegistry.get_instance().safety_critic
        result = await agent.critique(state)
        final_answer = result.get("final_answer", state.get("final_answer", ""))
        predicted_letter = result.get("predicted_letter", state.get("predicted_letter", "UNKNOWN"))
        safety_flags = result.get("safety_flags", state.get("safety_flags", []))

        trace_event = build_trace_event(
            state,
            section="final_output",
            event="safety_critic",
            node="safety_critic",
            data={
                "predicted_letter": predicted_letter,
                "final_answer_len": len(str(final_answer or "")),
                "safety_flags": safety_flags,
            },
            influence={"state_updates": list(result.keys())},
            attach_context=False,
        )
        trace_updates = build_trace_updates(state, [trace_event])
        result["trace_events"] = result.get("trace_events", []) + trace_updates.get("trace_events", [])
        if "trace_id" not in result:
            result["trace_id"] = trace_updates.get("trace_id")
        if "trace_created_at" not in result:
            result["trace_created_at"] = trace_updates.get("trace_created_at")
        return result
    except Exception as e:
        logger.error(f"safety_critic_node crashed: {e}", exc_info=True)
        return {
            "safety_flags": state.get("safety_flags", []) + ["node_crash"],
            "predicted_letter": state.get("predicted_letter", "UNKNOWN"),
        }