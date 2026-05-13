import logging
import asyncio
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.core.registry import ModelRegistry, safe_ainvoke
from src.prompts.templates import DIALECTICAL_SYNTHESIS_SYSTEM_PROMPT, DIALECTICAL_SYNTHESIS_HUMAN_PROMPT, with_json_system_suffix
from src.agents.registry import AgentRegistry
from src.state.state import GraphState
from src.epistemic.dempster_shafer import build_mass, combine_masses, normalize_triplet
from src.utils.epistemic_trace import build_trace_event, build_trace_updates

logger = logging.getLogger(__name__)

def _env_int(name: str, default: int, minimum: int = 1) -> int:
    raw_value = os.getenv(name, str(default)).strip()
    try:
        parsed = int(raw_value)
    except ValueError:
        parsed = default
    return max(minimum, parsed)


def _env_float(name: str, default: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    raw_value = os.getenv(name, str(default)).strip()
    try:
        parsed = float(raw_value)
    except ValueError:
        parsed = default
    return max(minimum, min(maximum, parsed))


_MAP_MAX_CHARS = _env_int("MRAGE_DIALECTIC_MAP_MAX_CHARS", 2000)
_MAP_MAX_SENTENCES = 2
_MAP_CONCURRENCY = _env_int("MRAGE_DIALECTIC_MAP_CONCURRENCY", 4)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_DIALECTIC_MAX_PAPERS_PER_SIDE = _env_int("MRAGE_DIALECTIC_MAX_PAPERS_PER_SIDE", 5)
_DIALECTIC_BYPASS_RPS_MIN = _env_float("MRAGE_DIALECTIC_BYPASS_RPS_MIN", 0.6)
_DIALECTIC_BYPASS_APP_MIN = _env_float("MRAGE_DIALECTIC_BYPASS_APP_MIN", 0.6)
_DIALECTIC_BYPASS_CONF_MIN = _env_float("MRAGE_DIALECTIC_BYPASS_CONF_MIN", 0.8)


class DialecticalSynthesisOutput(BaseModel):
    thesis_summary: str = Field(default="")
    antithesis_summary: str = Field(default="")
    convergence_points: str = Field(default="")
    divergence_points: str = Field(default="")
    synthesis: str = Field(default="")
    controversy_label: str = Field(default="EMERGING")
    confidence: float = Field(default=0.0)


_MAP_CONCLUSION_SYSTEM_PROMPT = (
    "You extract the single core clinical conclusion from an abstract. "
    "Return 1-2 sentences, no preamble, no bullets."
)
_MAP_CONCLUSION_HUMAN_PROMPT = "Abstract:\n{abstract}\n\nReturn only the core conclusion."


def _build_chain():
    llm = ModelRegistry.get_heavy_llm(temperature=0.0, json_mode=True)
    if llm is None:
        raise RuntimeError("dialectical_synthesis_node: Heavy LLM unavailable")
    parser = JsonOutputParser(pydantic_object=DialecticalSynthesisOutput)
    prompt = ChatPromptTemplate.from_messages([
        ("system", with_json_system_suffix(DIALECTICAL_SYNTHESIS_SYSTEM_PROMPT)),
        ("human", DIALECTICAL_SYNTHESIS_HUMAN_PROMPT),
    ])
    return prompt | llm | parser


def _build_map_chain():
    llm = ModelRegistry.get_light_llm(temperature=0.0, json_mode=False)
    if llm is None:
        return None
    prompt = ChatPromptTemplate.from_messages([
        ("system", _MAP_CONCLUSION_SYSTEM_PROMPT),
        ("human", _MAP_CONCLUSION_HUMAN_PROMPT),
    ])
    return prompt | llm


def _response_to_text(response: Any) -> str:
    if response is None:
        return ""
    if isinstance(response, str):
        return response
    if hasattr(response, "content"):
        return str(response.content)
    if isinstance(response, dict):
        return str(response.get("text", ""))
    return str(response)


def _truncate_for_map(text: str) -> str:
    cleaned = " ".join(str(text or "").split())
    if len(cleaned) <= _MAP_MAX_CHARS:
        return cleaned
    return cleaned[:_MAP_MAX_CHARS]


def _limit_sentences(text: str, max_sentences: int = _MAP_MAX_SENTENCES) -> str:
    cleaned = " ".join(str(text or "").split())
    if not cleaned:
        return ""
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(cleaned) if s.strip()]
    if not sentences:
        return cleaned
    return " ".join(sentences[:max_sentences])


async def _extract_core_conclusion(text: str, chain: Any | None) -> str:
    trimmed = _truncate_for_map(text)
    if not trimmed:
        return ""
    if chain is None:
        return _limit_sentences(trimmed)

    try:
        response = await safe_ainvoke(chain, {"abstract": trimmed})
    except Exception as exc:
        logger.debug("Abstract map failed, falling back to truncation: %s", exc)
        return _limit_sentences(trimmed)

    conclusion = _limit_sentences(_response_to_text(response))
    if not conclusion:
        return _limit_sentences(trimmed)
    return conclusion


def _looks_like_evidence_text(value: str) -> bool:
    lowered = value.lower()
    if lowered.startswith("title:") or "abstract:" in lowered:
        return True
    return not value.isdigit()


def _limit_evidence(items: List[str], limit: int) -> List[str]:
    if limit <= 0:
        return items
    return items[:limit]


async def _map_evidence_to_bullets(
    evidence: List[Tuple[Optional[str], Optional[str], str]],
    chain: Any | None,
) -> List[str]:
    semaphore = asyncio.Semaphore(_MAP_CONCURRENCY)

    async def _map_item(item: Tuple[Optional[str], Optional[str], str]) -> str | None:
        pmid, year, abstract = item
        async with semaphore:
            conclusion = await _extract_core_conclusion(abstract, chain)
        if not conclusion:
            return None
        if pmid:
            prefix = f"PMID:{pmid}"
            if year:
                prefix = f"{prefix} ({year})"
        else:
            prefix = "Provided evidence"
        return f"- {prefix}: {conclusion}"

    results = await asyncio.gather(*[_map_item(item) for item in evidence])
    return [result for result in results if result]


async def _fetch_evidence_block(
    client: Any | None,
    pmids: List[str],
    label: str,
    map_chain: Any | None,
) -> str:
    if not pmids:
        return f"{label}: none"

    raw_pmids: List[str] = []
    provided_texts: List[str] = []
    for item in pmids:
        text = str(item).strip()
        if not text:
            continue
        if _looks_like_evidence_text(text):
            provided_texts.append(text)
        else:
            raw_pmids.append(text)

    evidence_items: List[Tuple[Optional[str], Optional[str], str]] = []
    if provided_texts:
        for text in provided_texts:
            evidence_items.append((None, None, text))

    if raw_pmids:
        if client is None:
            logger.warning(
                "Dialectical synthesis: retriever client unavailable; skipping PubMed fetch for %s evidence.",
                label,
            )
            raw_pmids = []
        else:
            raw_articles = await client.efetch(raw_pmids)
            for article in raw_articles:
                if not article.abstract or not article.pmid:
                    continue
                year = str(article.year) if article.year is not None else None
                evidence_items.append((str(article.pmid), year, str(article.abstract)))

    if not evidence_items:
        return f"{label}: none"

    bullets = await _map_evidence_to_bullets(evidence_items, map_chain)
    if not bullets:
        return f"{label}: none"
    return f"{label}:\n" + "\n".join(bullets)


async def dialectical_synthesis_node(state: GraphState) -> Dict[str, Any]:
    """
    Reads:  state["thesis_docs"], state["antithesis_docs"],
            state["final_answer"], state["original_question"]
    Writes: state["dialectic_synthesis"]
    """
    thesis = state.get("thesis_docs", [])
    antithesis = state.get("antithesis_docs", [])
    final_answer = state.get("final_answer", "")
    question = state.get("original_question", "")
    # --- ADD THIS BELIEF REVISION FILTER ---
    overturned_pmids = set(state.get("overturned_pmids", []))

    if overturned_pmids:
        logger.info(f"Belief Revision: Filtering out {len(overturned_pmids)} overturned PMIDs from synthesis.")
        
        # Helper to check if a document/string contains any overturned PMID
        def _is_valid_evidence(doc: Any) -> bool:
            doc_str = str(doc)
            return not any(str(opmid) in doc_str for opmid in overturned_pmids)

        original_thesis_count = len(thesis)
        original_antithesis_count = len(antithesis)

        thesis = [doc for doc in thesis if _is_valid_evidence(doc)]
        antithesis = [doc for doc in antithesis if _is_valid_evidence(doc)]
        
        logger.info(
            f"Belief Revision complete. Thesis docs: {original_thesis_count} -> {len(thesis)}. "
            f"Antithesis docs: {original_antithesis_count} -> {len(antithesis)}."
        )
    # --- END BELIEF REVISION FILTER ---
    try:
        client = AgentRegistry.get_instance().retriever.client
    except Exception as exc:
        logger.warning(
            "Dialectical synthesis: retriever client unavailable; skipping PubMed fetch. error=%s",
            exc,
        )
        client = None

    thesis_count = len(thesis)
    antithesis_count = len(antithesis)

    evidence_polarity = state.get("evidence_polarity", {})
    polarity = "insufficient"
    confidence = 0.0
    if isinstance(evidence_polarity, dict):
        polarity = str(evidence_polarity.get("polarity", "insufficient")).lower()
        confidence = float(evidence_polarity.get("confidence", 0.0))

    rps_scores = state.get("rps_scores", []) or []
    if isinstance(rps_scores, list) and rps_scores:
        avg_rps = sum(float(score.get("final_score", score.get("rps_score", 0.0))) for score in rps_scores) / len(rps_scores)
    else:
        avg_rps = 0.0
    applicability_score = float(state.get("applicability_score", 0.0))

    strong_evidence = (
        (avg_rps >= _DIALECTIC_BYPASS_RPS_MIN and applicability_score >= _DIALECTIC_BYPASS_APP_MIN)
        or (polarity == "support" and confidence >= _DIALECTIC_BYPASS_CONF_MIN)
    )

    def _return_with_trace(payload: Dict[str, Any], note: str, label: str, has_conflict: bool) -> Dict[str, Any]:
        # --- Observability: Trace Enrichment ---
        # Capture current vs prior answer to detect synthesis influence
        prior_ans = state.get("final_answer", "")
        post_ans = payload.get("final_answer", prior_ans)
        synthesis_altered_answer = (prior_ans != post_ans)

        trace_event = build_trace_event(
            state,
            section="dialectical_synthesis",
            event="synthesis",
            node="dialectical_synthesis",
            data={
                "thesis_count": thesis_count,
                "antithesis_count": antithesis_count,
                "label": label,
                "has_conflict": has_conflict,
                "note": note,
                "synthesis_altered_answer": synthesis_altered_answer,
                "ds_mass_details": {
                    "thesis_mass": locals().get("thesis_mass"),
                    "antithesis_mass": locals().get("antithesis_mass"),
                    "combined_mass": locals().get("combined_mass")
                },
                "belief_revision_impact": {
                    "overturned_count": len(overturned_pmids),
                    "thesis_filtered": original_thesis_count - len(thesis) if 'original_thesis_count' in locals() else 0,
                    "antithesis_filtered": original_antithesis_count - len(antithesis) if 'original_antithesis_count' in locals() else 0
                }
            },
            influence={"state_updates": ["dialectic_synthesis", "final_answer", "predicted_letter"]},
            attach_context=False,
        )
        trace_updates = build_trace_updates(state, [trace_event])
        payload["trace_events"] = payload.get("trace_events", []) + trace_updates.get("trace_events", [])
        if "trace_id" not in payload:
            payload["trace_id"] = trace_updates.get("trace_id")
        if "trace_created_at" not in payload:
            payload["trace_created_at"] = trace_updates.get("trace_created_at")
        return payload


    if antithesis_count == 0 and strong_evidence:
        logger.info(
            "No antithesis docs found. Bypassing heavy dialectical LLM call | rps=%.2f app=%.2f pol=%s conf=%.2f",
            avg_rps,
            applicability_score,
            polarity,
            confidence,
        )
        synthesis = {
            "thesis_summary": "Evidence unanimously supports the primary claim.",
            "antithesis_summary": "No explicitly contradictory evidence retrieved.",
            "convergence_points": "General consensus in retrieved literature.",
            "divergence_points": "N/A",
            "synthesis": final_answer,
            "controversy_label": "SETTLED",
            "confidence": 0.95,
            "thesis_count": thesis_count,
            "antithesis_count": 0,
            "has_conflict": False,
        }
        return _return_with_trace(
            {
                "dialectic_synthesis": synthesis,
                "final_answer": final_answer,
            },
            "bypass_strong_evidence",
            "SETTLED",
            False,
        )

    try:
        map_chain = _build_map_chain()
        limited_thesis = _limit_evidence(thesis, _DIALECTIC_MAX_PAPERS_PER_SIDE)
        limited_antithesis = _limit_evidence(antithesis, _DIALECTIC_MAX_PAPERS_PER_SIDE)

        if len(limited_thesis) < thesis_count or len(limited_antithesis) < antithesis_count:
            logger.info(
                "Dialectical synthesis: limiting evidence (thesis %d->%d, antithesis %d->%d)",
                thesis_count,
                len(limited_thesis),
                antithesis_count,
                len(limited_antithesis),
            )

        thesis_evidence, antithesis_evidence = await asyncio.gather(
            _fetch_evidence_block(client, limited_thesis, "THESIS", map_chain),
            _fetch_evidence_block(client, limited_antithesis, "ANTITHESIS", map_chain),
        )

        # Epistemic plumbing: inject temporal context and a pragmatic
        # Dempster-Shafer summary into the LLM prompt so the LLM can
        # account for evidence evolution when synthesising.
        tcs_score = float(state.get("tcs_score", 0.0) or 0.0)
        temporal_conflicts = state.get("temporal_conflicts", []) or []
        if tcs_score > 0.3:
            conflict_context = (
                f"**Epistemic Alert**: Temporal Conflict Score (TCS) = {tcs_score:.2f}. "
                "This indicates a potential evolution or contradiction in evidence over time. "
                "Prefer newer findings where applicable."
            )
        else:
            conflict_context = ""

        # Pragmatic DS combination between thesis and antithesis aggregates.
        try:
            total = max(1, thesis_count + antithesis_count)
            t_ratio = float(thesis_count) / float(total)
            a_ratio = float(antithesis_count) / float(total)

            # 1. Base Probability Assignment (from raw retrieval ratios)
            base_thesis_support = t_ratio
            base_antithesis_refute = a_ratio

            # 2. Shafer's Discounting Rule via Reproducibility (RPS)
            # A low RPS means high uncertainty in the evidence itself.
            # Discount factor = (1 - avg_rps). We move the discounted mass into Uncertainty.
            reliability_discount = max(0.0, min(1.0, 1.0 - avg_rps))
            
            thesis_support_discounted = base_thesis_support * (1.0 - reliability_discount)
            thesis_uncertainty = 1.0 - thesis_support_discounted

            antithesis_refute_discounted = base_antithesis_refute * (1.0 - reliability_discount)
            antithesis_uncertainty = 1.0 - antithesis_refute_discounted

            thesis_mass = build_mass(thesis_support_discounted, 0.0)  # Thesis inherently supports
            antithesis_mass = build_mass(0.0, antithesis_refute_discounted)  # Antithesis inherently refutes

            # 3. Temporal Epistemic Discounting (TCS)
            # If temporal conflict is high, older/contradicted evidence (often in antithesis) 
            # must be mathematically decayed.
            if tcs_score > 0.0:
                # We decay the antithesis mass proportional to the conflict
                tcs_discount = min(0.9, tcs_score)
                a_sup = antithesis_mass.get("SUPPORT", 0.0)
                a_ref = antithesis_mass.get("REFUTE", 0.0) * (1.0 - tcs_discount)
                a_unc = min(1.0, antithesis_mass.get("UNCERTAIN", 0.0) + (antithesis_mass.get("REFUTE", 0.0) * tcs_discount))
                antithesis_mass = normalize_triplet(a_sup, a_ref, a_unc)

            combined_mass = combine_masses(thesis_mass, antithesis_mass)
            ds_summary = (
                f"Dempster-Shafer summary — SUPPORT={combined_mass['SUPPORT']:.2f}, "
                f"REFUTE={combined_mass['REFUTE']:.2f}, UNCERTAIN={combined_mass['UNCERTAIN']:.2f}."
            )

            # Derive a deterministic, DS-driven decision token so the
            # mathematical fusion drives the final choice (LLM only explains).
            try:
                ds_sup = float(combined_mass.get("SUPPORT", 0.0))
                ds_ref = float(combined_mass.get("REFUTE", 0.0))
            except Exception:
                ds_sup = 0.0
                ds_ref = 0.0

            if ds_sup > ds_ref:
                ds_predicted_letter = "A"
            elif ds_ref > ds_sup:
                ds_predicted_letter = "B"
            else:
                ds_predicted_letter = "UNKNOWN"
        except Exception as ds_exc:
            logger.debug("DS combination failed: %s", ds_exc)
            combined_mass = {"SUPPORT": 0.0, "REFUTE": 0.0, "UNCERTAIN": 1.0}
            ds_summary = "Dempster-Shafer summary unavailable"

        chain = _build_chain()
        result = await safe_ainvoke(chain,
            {
                "question": question,
                "thesis_evidence": thesis_evidence,
                "antithesis_evidence": antithesis_evidence,
                "prior_answer": final_answer,
                "temporal_context": conflict_context,
                "ds_summary": ds_summary,
                "ds_support": float(combined_mass.get("SUPPORT", 0.0)),
                "ds_refute": float(combined_mass.get("REFUTE", 0.0)),
                "ds_uncertain": float(combined_mass.get("UNCERTAIN", 1.0)),
                "ds_predicted_letter": ds_predicted_letter,
            }
        )

        label = str(result.get("controversy_label", "EMERGING")).upper()
        has_conflict = antithesis_count > 0 and label in {"CONTESTED", "OVERTURNED"}
        synthesis = {
            "thesis_summary": result.get("thesis_summary", ""),
            "antithesis_summary": result.get("antithesis_summary", ""),
            "convergence_points": result.get("convergence_points", ""),
            "divergence_points": result.get("divergence_points", ""),
            "synthesis": result.get("synthesis", final_answer),
            "controversy_label": label,
            "confidence": float(result.get("confidence", 0.0)),
            "thesis_count": thesis_count,
            "antithesis_count": antithesis_count,
            "has_conflict": has_conflict,
            "predicted_letter": ds_predicted_letter,
        }

        logger.info(
            "Dialectical synthesis generated: thesis=%s antithesis=%s label=%s",
            thesis_count,
            antithesis_count,
            label,
        )
        payload = {
            "dialectic_synthesis": synthesis,
            "final_answer": synthesis.get("synthesis", final_answer),
            "predicted_letter": synthesis.get("predicted_letter", ds_predicted_letter),
        }
        return _return_with_trace(payload, "llm_synthesis", label, has_conflict)

    except Exception as exc:
        logger.warning("Dialectical synthesis failed, returning conservative fallback: %s", exc)
        synthesis = {
                "thesis_summary": "",
                "antithesis_summary": "",
                "convergence_points": "",
                "divergence_points": "",
                "synthesis": final_answer,
                "controversy_label": "EMERGING",
                "confidence": 0.0,
                "thesis_count": thesis_count,
                "antithesis_count": antithesis_count,
                "has_conflict": antithesis_count > 0,
        }
        payload = {
            "dialectic_synthesis": synthesis,
            "final_answer": synthesis.get("synthesis", final_answer),
        }
        return _return_with_trace(payload, "fallback", "EMERGING", antithesis_count > 0)
