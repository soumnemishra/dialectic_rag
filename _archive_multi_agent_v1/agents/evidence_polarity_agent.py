import logging
import re
from typing import Dict, Any, List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential_jitter, RetryError

from src.state.state import GraphState, EvidencePolarity
from src.query_builder import parse_markdown_json
from src.prompts.templates import (
    EVIDENCE_POLARITY_SYSTEM_PROMPT,
    EVIDENCE_POLARITY_HUMAN_PROMPT,
    with_json_system_suffix,
)
from src.core.registry import ModelRegistry, safe_ainvoke
from src.agents.answer_utils import extract_final_answer_letter
from src.utils.epistemic_trace import build_trace_event, build_trace_updates

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  EvidencePolarityModule                                            #
# ------------------------------------------------------------------ #

class EvidencePolarityModule:
    """
    Detects the directional polarity of retrieved evidence relative to
    the original question: support / refute / mixed / insufficient.

    This runs AFTER the executor completes all plan steps, reading the
    current execution's extracted notes from GraphState.

    IMPORTANT: reads from state["step_notes"] and state["step_docs_ids"]
    — the CURRENT query's retrieved evidence — NOT from state["past_exp"]
    which holds data from previous FAILED retry attempts.

    The polarity result is written to state["evidence_polarity"] and is
    consumed by:
      - RagAgent: adds conflict note when polarity is "refute"/"mixed"
      - SafetyCriticAgent: enforces disclaimer when polarity is "refute"
      - SummaryAgent: flags contradictory evidence for "mixed"

    Failure is NON-BLOCKING — a polarity failure returns "insufficient"
    and lets the pipeline continue rather than aborting the query.
    """

    def __init__(self):
        self.llm = ModelRegistry.get_flash_llm(temperature=0.0, json_mode=True)

        # Fail loud at startup, not silently mid-query
        if self.llm is None:
            raise RuntimeError(
                "EvidencePolarityAgent: Flash LLM failed to load. "
                "Check Gemini configuration (GEMINI_MODEL_HEAVY/GEMINI_MODEL_LIGHT) or GOOGLE_API_KEY."
            )

        self.parser = JsonOutputParser(pydantic_object=EvidencePolarity)

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", with_json_system_suffix(EVIDENCE_POLARITY_SYSTEM_PROMPT)),
            ("human",  EVIDENCE_POLARITY_HUMAN_PROMPT),
        ])

        # Build chain ONCE in __init__ — not reconstructed per analyze() call
        self.chain = self.prompt | self.llm | self.parser
        self.raw_chain = self.prompt | self.llm

    # ---------------------------------------------------------------- #
    #  Private helpers                                                   #
    # ---------------------------------------------------------------- #

    @staticmethod
    def _normalize_letter(value: str) -> str:
        letter = str(value or "").strip().upper()
        return letter if letter in {"A", "B", "C", "D"} else "UNKNOWN"

    @staticmethod
    def _extract_mcq_options(text: str) -> Dict[str, str]:
        options: Dict[str, str] = {}
        if not text:
            return options
        for match in re.finditer(r"(?im)^\s*([A-D])\s*[:\.)]\s*(.+?)\s*$", str(text)):
            letter = match.group(1).upper()
            option_text = match.group(2).strip()
            if option_text:
                options[letter] = option_text
        return options

    def _resolve_predicted_option_text(self, state: GraphState, predicted_letter: str) -> str:
        mcq_options = state.get("mcq_options", "") or ""
        original_question = state.get("original_question", "") or ""
        if isinstance(mcq_options, dict):
            options = {str(k).upper(): str(v) for k, v in mcq_options.items() if v}
        else:
            options = self._extract_mcq_options(mcq_options) or self._extract_mcq_options(original_question)
        return options.get(predicted_letter, "")

    def _resolve_claimed_answer(self, state: GraphState, predicted_letter: str) -> str:
        option_text = self._resolve_predicted_option_text(state, predicted_letter)
        if option_text:
            return option_text
        return predicted_letter if predicted_letter != "UNKNOWN" else "UNKNOWN"

    def _resolve_predicted_letter(self, state: GraphState) -> str:
        primary = self._normalize_letter(state.get("predicted_letter", ""))
        if primary != "UNKNOWN":
            return primary

        plan_summary = state.get("plan_summary")
        if isinstance(plan_summary, dict):
            for key in (
                "mcq_letter",
                "predicted_letter",
                "final_decision",
                "answer",
                "final_answer",
                "summary",
                "analysis",
                "output",
            ):
                letter = extract_final_answer_letter(plan_summary.get(key), fallback="UNKNOWN")
                letter = self._normalize_letter(letter)
                if letter != "UNKNOWN":
                    return letter

        step_outputs = state.get("step_output", [])
        if isinstance(step_outputs, list):
            for output in reversed(step_outputs):
                if not isinstance(output, dict):
                    continue
                for key in (
                    "mcq_letter",
                    "predicted_letter",
                    "answer",
                    "final_answer",
                    "summary",
                    "analysis",
                ):
                    letter = extract_final_answer_letter(output.get(key), fallback="UNKNOWN")
                    letter = self._normalize_letter(letter)
                    if letter != "UNKNOWN":
                        return letter

        final_answer = state.get("final_answer", "")
        letter = extract_final_answer_letter(final_answer, fallback="UNKNOWN")
        return self._normalize_letter(letter)

    def _format_evidence(self, state: GraphState) -> str:
        """
        Build a readable evidence string from the CURRENT execution's
        retrieved notes and document IDs.

        Why NOT past_exp:
            state["past_exp"] stores results from PREVIOUS FAILED query
            attempts — it is the planner's retry memory. On the first
            attempt it is always empty.

            The current execution's evidence lives in:
              state["step_notes"]     — ExtractorAgent output this query
              state["step_docs_ids"]  — PMIDs retrieved this query
              state["step_output"]    — RagAgent answers this query

        Returns a formatted string, or a clear "no evidence" message
        so the LLM doesn't hallucinate a polarity from nothing.
        #take the step notes and step_docs_ids
        """
        step_notes   = state.get("step_notes",    [])
        step_doc_ids = state.get("step_docs_ids", [])
        step_outputs = state.get("step_output",   [])

        if not step_notes and not step_outputs:
            return "No evidence retrieved in current execution."

        lines = []
        count = 1

        # Select up to N most relevant notes (by extracted relevance, or fast similarity)
        try:
            note_limit = int(__import__("os").getenv("MRAGE_EVIDENCE_NOTE_LIMIT", "20"))
        except Exception:
            note_limit = 20

        candidates = []
        question_text = state.get("original_question", "") or ""

        # Attempt to use cross-encoder if available for fast relevance scoring
        try:
            reranker_model = None
            from src.agents.registry import AgentRegistry
            reranker = AgentRegistry.get_instance().retriever.reranker
            reranker_model = getattr(reranker, "_model", None)
        except Exception:
            reranker_model = None

        model_pairs: List[List[str]] = []
        model_indices: List[int] = []

        for i, note in enumerate(step_notes):
            if not note or str(note).strip() == "":
                continue
            doc_ids = step_doc_ids[i] if i < len(step_doc_ids) else []
            note_text = str(note).strip()
            score = None

            if isinstance(note, dict):
                if note.get("relevance") is not None:
                    try:
                        score = float(note.get("relevance"))
                    except Exception:
                        score = None
                elif note.get("reliability") is not None:
                    rel = str(note.get("reliability", "")).lower()
                    if rel == "high":
                        score = 0.9
                    elif rel == "medium":
                        score = 0.6
                    elif rel == "low":
                        score = 0.3

            # Defer cross-encoder scoring for batching
            if score is None and reranker_model is not None:
                model_pairs.append([question_text, note_text])
                model_indices.append(len(candidates))

            candidates.append({
                "idx": i,
                "doc_ids": doc_ids,
                "text": note_text,
                "score": float(score) if score is not None else None,
            })

        if model_pairs and reranker_model is not None:
            try:
                raw_scores = reranker_model.predict(model_pairs)
                if raw_scores is not None:
                    for local_idx, raw in enumerate(raw_scores):
                        if local_idx >= len(model_indices):
                            break
                        candidate_idx = model_indices[local_idx]
                        try:
                            candidates[candidate_idx]["score"] = float(raw)
                        except Exception:
                            candidates[candidate_idx]["score"] = None
            except Exception:
                logger.debug("EvidencePolarityAgent: batched reranker scoring failed; falling back to heuristics.")

        # Fallback token-overlap heuristic for any missing scores
        for candidate in candidates:
            if candidate.get("score") is not None:
                continue
            note_text = candidate.get("text", "")
            q_tokens = set(re.findall(r"\w+", question_text.lower()))
            n_tokens = set(re.findall(r"\w+", note_text.lower()))
            if q_tokens and n_tokens:
                score = len(q_tokens & n_tokens) / max(1, min(len(q_tokens), len(n_tokens)))
            else:
                score = 0.0
            candidate["score"] = float(score)

        # Sort candidates by score desc and pick top-K
        candidates.sort(key=lambda x: x["score"], reverse=True)
        selected = candidates[:note_limit]

        if len(candidates) > note_limit:
            logger.warning(
                "EvidencePolarityAgent: truncated evidence notes from %d -> %d (MRAGE_EVIDENCE_NOTE_LIMIT=%d)",
                len(candidates),
                len(selected),
                note_limit,
            )

        # Build lines from selected notes
        for c in selected:
            doc_ids = c["doc_ids"]
            if isinstance(doc_ids, list) and doc_ids:
                ids_str = f" (PMIDs: {', '.join(str(d) for d in doc_ids)})"
            else:
                ids_str = ""
            note_text = c["text"][:1000]
            lines.append(f"Evidence item {count}{ids_str}:\n{note_text}")
            count += 1

        # Also include step answers as supporting context
        answer_count = 0
        for i, output in enumerate(step_outputs):
            if isinstance(output, dict) and not output.get("is_error", False):
                answer = output.get("answer", "")
                if answer and answer.strip():
                    lines.append(f"Step {i + 1} answer:\n{answer[:300]}")
                    answer_count += 1
                    if answer_count >= 5:
                        break

        if not lines:
            return "No relevant evidence text found in current execution."

        return "\n\n".join(lines)

    @retry(stop=stop_after_attempt(2), wait=wait_exponential_jitter(initial=0.5, max=6))
    async def _invoke_chain(self, inputs: dict) -> dict:
        """
        Runs the chain with up to 2 retries on transient failures.
        Separated into its own method for independent testability.
        """
        try:
            return await safe_ainvoke(self.chain, inputs)
        except Exception:
            raw_result = await safe_ainvoke(self.raw_chain, inputs)
            raw_text = raw_result.content if hasattr(raw_result, "content") else str(raw_result)
            return parse_markdown_json(raw_text)

    # ---------------------------------------------------------------- #
    #  Public interface                                                  #
    # ---------------------------------------------------------------- #

    async def analyze(self, state: GraphState) -> Dict[str, Any]:
        """
        Analyse the current execution's retrieved evidence and classify
        its polarity relative to the original question.

        Reads from:
            state["original_question"]  — the query being answered
            state["step_notes"]         — current execution extracted notes
            state["step_docs_ids"]      — current execution PMIDs
            state["step_output"]        — current execution step answers

        Writes to GraphState:
            evidence_polarity: {
                "polarity":   "support" | "refute" | "mixed" | "insufficient",
                "confidence": float 0.0–1.0,
                "reasoning":  str — one-sentence explanation
            }

        Downstream consumers of evidence_polarity:
            - RagAgent QA_HUMAN_PROMPT: {evidence_polarity} placeholder
            - SafetyCriticAgent: adds disclaimer when polarity = "refute"
            - SummaryAgent: flags conflict when polarity = "mixed"
        """
        try:
            question        = state["original_question"]
            proposed_answer = state.get("final_answer", "") or state.get("answer", "")
            predicted_letter = self._resolve_predicted_letter(state)
            predicted_option_text = self._resolve_claimed_answer(state, predicted_letter)
            evidence_text  = self._format_evidence(state)

            logger.info(
                f"Analysing evidence polarity for: '{question[:80]}...' "
                f"({len(state.get('step_notes', []))} step notes)"
            )

            try:
                result = await self._invoke_chain({
                    "question": question,
                    "proposed_answer": proposed_answer,
                    "predicted_letter": predicted_letter,
                    "predicted_option_text": predicted_option_text,
                    "claimed_answer": predicted_option_text,
                    "evidence": evidence_text,
                })
            except RetryError as e:
                raise ValueError(f"Polarity chain failed after 2 attempts: {e}") from e

            polarity   = result.get("polarity",   "insufficient")
            confidence = float(result.get("confidence", 0.0))
            reasoning  = result.get("reasoning",  "")

            # Validate polarity value — LLM sometimes returns unexpected strings
            valid_polarities = {"strong_support", "weak_support", "refute", "insufficient"}
            if polarity not in valid_polarities:
                logger.warning(
                    f"LLM returned unexpected polarity '{polarity}' — "
                    f"defaulting to 'insufficient'"
                )
                polarity = "insufficient"

            logger.info(
                f"Evidence polarity: {polarity} "
                f"(confidence={confidence:.2f}) — {reasoning}"
            )

            # --- REMEDIATION Rule 7: Cross-Check committed steps ---
            step_outputs = state.get("step_output", [])
            committed_letters = set()
            for out in step_outputs:
                if isinstance(out, dict) and not out.get("is_error"):
                    letter = extract_final_answer_letter(
                        out.get("predicted_letter") or out.get("answer") or out.get("predicted") or "",
                        fallback="UNKNOWN"
                    )
                    if letter in {"A", "B", "C", "D"}:
                        committed_letters.add(letter)
            
            if len(committed_letters) > 1:
                logger.warning(f"Conflicting committed step answers detected: {committed_letters}. Forcing 'insufficient' (conflicting) polarity.")
                polarity = "insufficient"
                reasoning = f"Conflicting committed step answers detected among sub-steps ({', '.join(sorted(committed_letters))})."
                confidence = 1.0

            return {  
                "evidence_polarity": {
                    "polarity":   polarity,
                    "confidence": confidence,
                    "reasoning":  reasoning,
                }
            }

        except Exception as e:
            logger.error(f"Evidence polarity analysis failed: {e}", exc_info=True)
            # NON-BLOCKING — return safe default, pipeline continues
            return {
                "evidence_polarity": {
                    "polarity":   "insufficient",
                    "confidence": 0.0,
                    "reasoning":  f"Analysis failed: {e}",
                }
            }


# ------------------------------------------------------------------ #
#  LangGraph node wrapper                                             #
# ------------------------------------------------------------------ #

from src.agents.registry import AgentRegistry


async def evidence_polarity_node(state: GraphState) -> Dict[str, Any]:
    """
    Thin wrapper called by the LangGraph StateGraph.
    Polarity is computed AFTER primary retrieval completes,
    so step_notes and step_docs_ids are fully populated.
    """
    try:
        module = AgentRegistry.get_instance().evidence_polarity
        result = await module.analyze(state)

        polarity = result.get("evidence_polarity", {}).get("polarity", "insufficient")
        confidence = float(result.get("evidence_polarity", {}).get("confidence", 0.0))
        step_notes = state.get("step_notes", []) or []
        step_docs = state.get("step_docs_ids", []) or []

        trace_event = build_trace_event(
            state,
            section="evidence_analysis",
            event="polarity",
            node="evidence_polarity",
            data={
                "polarity": polarity,
                "confidence": confidence,
                "reasoning": result.get("evidence_polarity", {}).get("reasoning", ""),
                "evidence_note_count": len(step_notes),
                "evidence_doc_id_groups": len(step_docs),
            },
            influence={"state_updates": ["evidence_polarity"]},
            attach_context=False,
        )
        trace_updates = build_trace_updates(state, [trace_event])

        merged = dict(result)
        merged["trace_events"] = merged.get("trace_events", []) + trace_updates.get("trace_events", [])
        if "trace_id" not in merged:
            merged["trace_id"] = trace_updates.get("trace_id")
        if "trace_created_at" not in merged:
            merged["trace_created_at"] = trace_updates.get("trace_created_at")

        return merged
    except Exception as e:
        logger.error(f"evidence_polarity_node crashed: {e}", exc_info=True)
        # Non-blocking — pipeline must not abort because polarity failed
        return {
            "evidence_polarity": {
                "polarity":   "insufficient",
                "confidence": 0.0,
                "reasoning":  f"Node crash: {e}",
            }
        }