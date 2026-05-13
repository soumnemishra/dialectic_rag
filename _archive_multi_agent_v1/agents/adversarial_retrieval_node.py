import logging
import os
from typing import Any, Dict, List

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.core.registry import ModelRegistry, safe_ainvoke
from src.prompts.templates import ADVERSARIAL_QUERY_SYSTEM_PROMPT, ADVERSARIAL_QUERY_HUMAN_PROMPT, with_json_system_suffix
from src.state.state import GraphState
from src.agents.registry import AgentRegistry
from src.utils.epistemic_trace import build_trace_event, build_trace_updates

logger = logging.getLogger(__name__)


class AdversarialQueryOutput(BaseModel):
    adversarial_query: str = Field(default="")
    reasoning: str = Field(default="")


def _build_chain():
    llm = ModelRegistry.get_flash_llm(temperature=0.0, json_mode=True)
    if llm is None:
        raise RuntimeError("adversarial_retrieval_node: Flash LLM unavailable")
    parser = JsonOutputParser(pydantic_object=AdversarialQueryOutput)
    prompt = ChatPromptTemplate.from_messages([
        ("system", with_json_system_suffix(ADVERSARIAL_QUERY_SYSTEM_PROMPT)),
        ("human", ADVERSARIAL_QUERY_HUMAN_PROMPT),
    ])
    return prompt | llm | parser


def _unique_pmids(step_docs_ids: List[Any]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for item in step_docs_ids:
        ids = item if isinstance(item, list) else [item]
        for pmid in ids:
            value = str(pmid)
            if value and value not in seen:
                seen.add(value)
                ordered.append(value)
    return ordered


def _format_antithesis_text(doc: Any) -> str:
    title = str(getattr(doc, "title", None) or "No Title")
    abstract = str(getattr(doc, "abstract", None) or "No Abstract")
    return f"Title: {title}\nAbstract: {abstract}"


def _select_hypothesis(state: GraphState) -> str:
    use_final = os.getenv("MRAGE_ADVERSARIAL_USE_FINAL_ANSWER", "1").strip().lower() not in {"0", "false", "no"}
    final_answer = str(state.get("final_answer", "") or "").strip()
    if use_final and final_answer:
        return final_answer

    step_outputs = state.get("step_output", [])
    if not step_outputs:
        return ""

    last_output = step_outputs[-1]
    if isinstance(last_output, list) and last_output:
        last_output = last_output[0]
    if isinstance(last_output, dict):
        return str(last_output.get("answer", "") or last_output.get("summary", "") or "").strip()
    return str(last_output).strip()


def _is_negative_text(text: str) -> bool | None:
    """Heuristic polarity detector: returns True if text is negative,
    False if positive, or None if uncertain.

    This is intentionally conservative and only used for a polarity-aware
    fallback when the LLM fails to produce an adversarial query.
    """
    if not text:
        return None
    t = text.lower()
    negative_keywords = {
        "not", "no", "ineffective", "harm", "harmful", "adverse",
        "mortality", "failure", "unsafe", "risk", "risks", "worse",
        "decreased", "decline",
    }
    positive_keywords = {
        "effective", "efficacy", "beneficial", "benefit", "improve",
        "improves", "safe", "survival", "reduced", "reduces", "support",
        "supports", "significant", "statistically significant", "better",
    }
    neg_count = sum(1 for kw in negative_keywords if kw in t)
    pos_count = sum(1 for kw in positive_keywords if kw in t)
    if neg_count > pos_count:
        return True
    if pos_count > neg_count:
        return False
    return None


async def adversarial_retrieval_node(state: GraphState) -> Dict[str, Any]:
    """
    Reads:  state["original_question"], state["step_docs_ids"], state["step_output"], state["final_answer"]
    Writes: state["antithesis_docs"], state["thesis_docs"]
    """
    all_prior_pmids = _unique_pmids(state.get("step_docs_ids", []))
    thesis_docs = all_prior_pmids[:10]
    question = state.get("original_question", "")
    antithesis_docs: List[str] = []

    if not question.strip():
        return {"thesis_docs": thesis_docs, "antithesis_docs": []}

    evidence_intent = state.get("intent", "diagnosis")
    current_hypothesis = _select_hypothesis(state)

    adversarial_query = ""
     
    try:
        chain = _build_chain()
        result = await safe_ainvoke(chain, {
            "original_query": question,
            "intent": evidence_intent,
            "current_hypothesis": current_hypothesis
        })
        adversarial_query = str(result.get("adversarial_query", "")).strip()
        
        if not adversarial_query:
            logger.warning("Adversarial query generation returned empty query. Applying polarity-aware fallback.")
            neg_terms = "adverse effects OR mortality OR failure OR controversy"
            pos_terms = "safety OR efficacy OR survival OR benign"

            # Determine polarity from the selected hypothesis if available,
            # otherwise fall back to the original question. If polarity is
            # ambiguous, include both sides to avoid creating an echo chamber.
            polarity_source = current_hypothesis if current_hypothesis else question
            is_negative = _is_negative_text(polarity_source)
            if is_negative is None:
                adversarial_query = f"{question} AND ({neg_terms} OR {pos_terms})"
            elif is_negative:
                # Hypothesis is negative — search for positive/benign evidence
                adversarial_query = f"{question} AND ({pos_terms})"
            else:
                # Hypothesis is positive — search for negative/adverse evidence
                adversarial_query = f"{question} AND ({neg_terms})"

        try:
            client = AgentRegistry.get_instance().retriever.client
        except Exception as exc:
            logger.warning(
                "Adversarial retrieval: retriever client unavailable; skipping PubMed search. error=%s",
                exc,
            )
            return {"thesis_docs": thesis_docs, "antithesis_docs": []}

        papers = await client.search(adversarial_query, max_results=10)
        thesis_set = set(all_prior_pmids)
        raw_antithesis: List[str] = []
        for doc in papers:
            pmid = str(getattr(doc, "pmid", "") or "").strip()
            if not pmid or pmid in thesis_set:
                continue
            formatted = _format_antithesis_text(doc)
            if formatted:
                raw_antithesis.append(formatted)

        antithesis_docs = raw_antithesis[:5]

    except Exception as exc:
        logger.warning("Adversarial retrieval failed, continuing without antithesis docs: %s", exc)
        antithesis_docs = []

    logger.info(
        "Adversarial retrieval split complete: thesis=%s antithesis=%s",
        len(thesis_docs), len(antithesis_docs),
    )
    
    step_idx = len(state.get("step_output", []))
    logger.info(f"Dialectical retrieval executed for step {step_idx} | found {len(antithesis_docs)} antithetical documents")

    payload = {
        "thesis_docs": thesis_docs,
        "antithesis_docs": antithesis_docs,
        "dialectical_retrieval_done": True,
    }
    trace_event = build_trace_event(
        state,
        section="dialectical_retrieval",
        event="adversarial_retrieval",
        node="adversarial_retrieval",
        data={
            "adversarial_query": adversarial_query,
            "thesis_count": len(thesis_docs),
            "antithesis_count": len(antithesis_docs),
        },
        influence={"state_updates": ["thesis_docs", "antithesis_docs"]},
        attach_context=False,
    )
    trace_updates = build_trace_updates(state, [trace_event])
    payload["trace_events"] = payload.get("trace_events", []) + trace_updates.get("trace_events", [])
    if "trace_id" not in payload:
        payload["trace_id"] = trace_updates.get("trace_id")
    if "trace_created_at" not in payload:
        payload["trace_created_at"] = trace_updates.get("trace_created_at")
    return payload
