"""Prompt-based temporal conflict classification for biomedical abstracts."""

import asyncio
import logging
import math
import re
from typing import Any, Dict, List, Tuple

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.core.registry import ModelRegistry, safe_ainvoke
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential_jitter,
    retry_if_exception_type,
)
from src.config import settings
from src.prompts.templates import TEMPORAL_CONFLICT_SYSTEM_PROMPT, TEMPORAL_CONFLICT_HUMAN_PROMPT, with_json_system_suffix
from src.pubmed_client import PubMedClient, RawArticle
from src.state.state import GraphState
from src.utils.epistemic_config import get_epistemic_setting
from src.agents.registry import AgentRegistry
from src.utils.epistemic_trace import build_trace_event, build_trace_updates


def _get_nli_agent():
    """Backward-compatible accessor used by tests and callers.

    Returns an NLI agent instance from the AgentRegistry, or None
    if initialization failed.
    """
    try:
        return AgentRegistry.get_instance().nli_agent
    except Exception as e:
        logger.debug("NLI agent unavailable from registry: %s", e)
        return None

logger = logging.getLogger(__name__)




TEMPORAL_LAMBDA = 0.3
MIN_YEAR_GAP = int(get_epistemic_setting(
    "temporal_conflict.min_year_gap",
    2,
    env_var="MRAGE_TCS_MIN_YEAR_GAP",
))
MAX_PAPERS = int(get_epistemic_setting(
    "temporal_conflict.max_papers",
    30,
    env_var="MRAGE_TCS_MAX_PAPERS",
))
# Use the centralized settings instance so MRAGE_* env vars and pydantic
# defaults are respected consistently across the codebase.
MAX_PAIRS_TO_SCORE = int(getattr(settings, "MRAGE_TCS_MAX_PAIRS", 20))

# Concurrency for pair classification — allow global LLM concurrency to
# dictate TCS parallelism, but tighten under FAST_EPISTEMIC.
DEFAULT_TCS_CONCURRENCY = int(getattr(settings, "MRAGE_LLM_MAX_CONCURRENCY", 5))
if settings.FAST_EPISTEMIC:
    MAX_CONCURRENT_PAIRS = max(1, min(DEFAULT_TCS_CONCURRENCY, int(getattr(settings, "MRAGE_FAST_EPISTEMIC_CAP", 2))))
else:
    MAX_CONCURRENT_PAIRS = DEFAULT_TCS_CONCURRENCY


class ConflictResult(BaseModel):
    direction: str = Field(description="CONTRADICT | SUPPORT | NEUTRAL")
    confidence: float = Field(default=0.0)
    older_claim: str = Field(default="")
    newer_claim: str = Field(default="")
    reasoning: str = Field(default="")


def _build_temporal_chain():
    llm = ModelRegistry.get_flash_llm(temperature=0.0, json_mode=True)
    if llm is None:
        raise RuntimeError("temporal_conflict_node: Flash LLM unavailable")
    parser = JsonOutputParser(pydantic_object=ConflictResult)
    prompt = ChatPromptTemplate.from_messages([
        ("system", with_json_system_suffix(TEMPORAL_CONFLICT_SYSTEM_PROMPT)),
        ("human", TEMPORAL_CONFLICT_HUMAN_PROMPT),
    ])
    return prompt | llm | parser


def _flatten_unique_pmids(step_docs_ids: List[Any]) -> List[str]:
    ordered: List[str] = []
    seen = set()
    for item in step_docs_ids:
        ids = item if isinstance(item, list) else []
        for pmid in ids:
            value = str(pmid)
            if value and value not in seen:
                seen.add(value)
                ordered.append(value)
    return ordered


def _limit_temporal_pairs(
    pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]],
    total_docs: int
) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    # Scale dynamically based on the total number of retrieved documents
    dynamic_limit = max(MAX_PAIRS_TO_SCORE, total_docs)
    if len(pairs) <= dynamic_limit:
        return pairs

    pairs.sort(key=lambda item: item[1]["year"] - item[0]["year"], reverse=True)
    return pairs[:dynamic_limit]


async def _fetch_papers_with_year(client: PubMedClient, pmids: List[str]) -> List[Dict[str, Any]]:
    papers: List[Dict[str, Any]] = []
    try:
        raw_articles: List[RawArticle] = await client.efetch(pmids)
        for article in raw_articles:
            if article.year and article.abstract and article.pmid:
                papers.append({
                    "pmid": str(article.pmid),
                    "year": int(article.year),
                    "abstract": str(article.abstract)[:1500],
                })
    except Exception as exc:
        logger.error("_fetch_papers_with_year failed: %s", exc)
    return papers


async def _classify_pair(chain, older: Dict[str, Any], newer: Dict[str, Any]) -> Dict[str, Any]:
    """Classify a paper pair with prompt-based temporal conflict classification.

    After LLM classification, optionally calls NliAgent to cross-check the
    older vs newer claims. Agreement boosts confidence; disagreement reduces it.
    """
    try:
        # Allow longer timeout for expensive temporal comparisons so
        # Tenacity retries in the provider client have time to succeed.
        result = await safe_ainvoke(
            chain,
            {
                "pmid_older": older["pmid"],
                "year_older": older["year"],
                "abstract_older": older["abstract"],
                "pmid_newer": newer["pmid"],
                "year_newer": newer["year"],
                "abstract_newer": newer["abstract"],
            },
            timeout=300.0,
        )
    except Exception as exc:
        # Log the transient error and re-raise so the tenacity retry wrapper
        # can attempt the call again. After retries exhaust, the caller
        # (_bounded_classify) will catch the exception and return a neutral
        # result to avoid crashing the pipeline.
        logger.warning(
            "Temporal conflict classification failed for %s vs %s: %s: %s",
            older.get("pmid", "unknown"),
            newer.get("pmid", "unknown"),
            type(exc).__name__,
            exc,
            exc_info=True,
        )
        raise

    # ── NLI cross-check (optional enhancement) ─────────────────────
    try:
        # Use the registry singleton so the NLI model is cached and only
        # initialised once per process. Keep a backward-compatible accessor
        # so unit tests can patch `_get_nli_agent`.
        nli_agent = _get_nli_agent()
        older_claim = str(result.get("older_claim", "")).strip()
        newer_claim = str(result.get("newer_claim", "")).strip()
        if nli_agent and older_claim and newer_claim:
            nli_result = await nli_agent.classify(older_claim, newer_claim)
            nli_label = nli_result.get("label", "NEUTRAL")
            llm_direction = result.get("direction", "NEUTRAL")
            conf = float(result.get("confidence", 0.0))

            if nli_label == "CONTRADICTION" and llm_direction == "CONTRADICT":
                result["confidence"] = round(min(1.0, conf * 1.15), 3)
                logger.debug(
                    "NLI agrees with CONTRADICT for %s vs %s; confidence %.3f → %.3f",
                    older.get("pmid"), newer.get("pmid"), conf, result["confidence"],
                )
            elif nli_label == "ENTAILMENT" and llm_direction == "CONTRADICT":
                result["confidence"] = round(max(0.0, conf * 0.85), 3)
                logger.debug(
                    "NLI disagrees (ENTAILMENT) with CONTRADICT for %s vs %s; confidence %.3f → %.3f",
                    older.get("pmid"), newer.get("pmid"), conf, result["confidence"],
                )
    except Exception as nli_exc:
        logger.debug("NLI cross-check skipped: %s", nli_exc)

    return result


# Wrap the pair classifier with a lightweight retry to overcome transient
# rate-limited errors from the LLM service. Retry on transient network
# or timeout errors up to 3 attempts with jittered exponential backoff.
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential_jitter(initial=2, max=10),
    retry=retry_if_exception_type((asyncio.TimeoutError, Exception)),
)
async def _classify_pair_with_retry(chain, older: Dict[str, Any], newer: Dict[str, Any]) -> Dict[str, Any]:
    return await _classify_pair(chain, older, newer)


async def temporal_conflict_node(state: GraphState) -> Dict[str, Any]:
    """
    Reads:  state["step_docs_ids"]
    Writes: state["tcs_score"], state["temporal_conflicts"]
    """
    def _return_with_trace(payload: Dict[str, Any], note: str, pmid_count: int, pair_count: int = 0) -> Dict[str, Any]:
        trace_event = build_trace_event(
            state,
            section="evidence_analysis",
            event="temporal_conflict",
            node="temporal_conflict",
            data={
                "tcs_score": payload.get("tcs_score", 0.0),
                "pmid_count": pmid_count,
                "pair_count": pair_count,
                "contradiction_count": len(payload.get("temporal_conflicts", []) or []),
                "overturned_pmids": len(payload.get("overturned_pmids", []) or []),
                "note": note,
            },
            influence={"state_updates": list(payload.keys())},
            attach_context=False,
        )
        trace_updates = build_trace_updates(state, [trace_event])
        merged = dict(payload)
        merged["trace_events"] = merged.get("trace_events", []) + trace_updates.get("trace_events", [])
        if "trace_id" not in merged:
            merged["trace_id"] = trace_updates.get("trace_id")
        if "trace_created_at" not in merged:
            merged["trace_created_at"] = trace_updates.get("trace_created_at")
        return merged

    all_pmids = _flatten_unique_pmids(state.get("step_docs_ids", []))[:MAX_PAPERS]
    if len(all_pmids) < 2:
        logger.info("temporal_conflict_node: fewer than 2 papers - TCS=null")
        return _return_with_trace({"tcs_score": None, "temporal_conflicts": []}, "insufficient_pmids", len(all_pmids))

    client = PubMedClient(filter_humans=False, filter_recent_years=None, filter_study_types=False)
    chain = _build_temporal_chain()
    papers = await _fetch_papers_with_year(client, all_pmids)
    if len(papers) < 2:
        return _return_with_trace({"tcs_score": None, "temporal_conflicts": []}, "insufficient_papers", len(all_pmids))

    # Filter out papers with missing or None years
    papers_with_year = [p for p in papers if p.get("year") is not None]
    if len(papers_with_year) < 2:
        return _return_with_trace({"tcs_score": None, "temporal_conflicts": []}, "insufficient_year_metadata", len(all_pmids))

    papers_sorted = sorted(papers_with_year, key=lambda item: item["year"])

    if not papers_sorted:
        return _return_with_trace({"tcs_score": None, "temporal_conflicts": []}, "no_sorted_papers", len(all_pmids))

    current_year = max([p["year"] for p in papers_sorted])
    recent_threshold = current_year - 5

    recent_papers = [p for p in papers_sorted if p["year"] > recent_threshold]
    older_papers = [p for p in papers_sorted if p["year"] <= recent_threshold]

    # If all papers are recent or all are old, split them by median
    if not recent_papers or not older_papers:
        mid = len(papers_sorted) // 2
        older_papers = papers_sorted[:mid]
        recent_papers = papers_sorted[mid:]

    pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for older in older_papers:
        for newer in recent_papers:
            if newer["pmid"] == older["pmid"]:
                continue
            if (newer["year"] - older["year"]) < MIN_YEAR_GAP:
                continue
            pairs.append((older, newer))
    if not pairs:
        return _return_with_trace({"tcs_score": None, "temporal_conflicts": []}, "no_pairs", len(all_pmids))

    pairs = _limit_temporal_pairs(pairs, len(all_pmids))

    logger.info(f"temporal_conflict_node: classifying {len(pairs)} paper pairs")
    sem = asyncio.Semaphore(MAX_CONCURRENT_PAIRS)

    async def _bounded_classify(older: Dict[str, Any], newer: Dict[str, Any]) -> Dict[str, Any]:
        async with sem:
            try:
                return await _classify_pair_with_retry(chain, older, newer)
            except Exception as exc:
                logger.warning("Temporal pair classify (with retry) failed: %s", exc, exc_info=True)
                # Return neutral result on persistent failure
                return {
                    "direction": "NEUTRAL",
                    "confidence": 0.0,
                    "older_claim": "",
                    "newer_claim": "",
                    "reasoning": "Classification failed after retries",
                }

    results = await asyncio.gather(
        *[_bounded_classify(older, newer) for older, newer in pairs],
        return_exceptions=True,
    )

    total_weight = 0.0
    conflict_sum = 0.0
    conflicts_out: List[Dict[str, Any]] = []

    for (older, newer), result in zip(pairs, results):
        if isinstance(result, Exception):
            logger.warning("Pair classification failed: %s", result, exc_info=True)
            continue

        try:
            year_newer_raw = newer.get("year", "")
            year_older_raw = older.get("year", "")
            newer_match = re.search(r"\d{4}", str(year_newer_raw))
            older_match = re.search(r"\d{4}", str(year_older_raw))
            if not newer_match or not older_match:
                continue
            year_newer = int(newer_match.group(0))
            year_older = int(older_match.group(0))
            if year_newer <= 0 or year_older <= 0:
                continue
            delta_year = year_newer - year_older
        except (TypeError, ValueError):
            logger.warning("Invalid year metadata in pair; skipping temporal comparison.")
            continue
        weight = 1.0 / (1.0 + math.exp(-TEMPORAL_LAMBDA * delta_year))
        total_weight += weight

        if result.get("direction") == "CONTRADICT":
            conf = float(result.get("confidence", 0.5))
            conflict_sum += weight * conf
            conflicts_out.append({
                "pmid_a": older["pmid"],
                "pmid_b": newer["pmid"],
                "year_a": year_older,
                "year_b": year_newer,
                "delta_year": delta_year,
                "direction": "CONTRADICT",
                "confidence": conf,
                "older_claim": result.get("older_claim", ""),
                "newer_claim": result.get("newer_claim", ""),
                "reasoning": result.get("reasoning", ""),
                "weight": round(weight, 3),
            })

    tcs = round(conflict_sum / total_weight, 3) if total_weight > 0 else None

    # Actively identify older PMIDs that have been confidently overturned
    # for temporal belief revision.
    overturned_pmids: List[str] = []
    for conflict in conflicts_out:
        try:
            if conflict.get("direction") == "CONTRADICT" and float(conflict.get("confidence", 0.0)) >= 0.85:
                # If NLI and LLM strongly agree the newer paper contradicts the older,
                # the older paper's claim is considered overturned.
                overturned_pmids.append(conflict.get("pmid_a"))
        except Exception:
            continue

    logger.info(
        f"temporal_conflict_node: TCS={tcs:.3f} | "
        f"{len(conflicts_out)} contradicting pairs / {len(pairs)} total | "
        f"Overturned PMIDs for revision: {len(overturned_pmids)}"
    )

    payload = {
        "tcs_score": tcs,
        "temporal_conflicts": conflicts_out,
        "overturned_pmids": list(set(overturned_pmids)),
    }
    return _return_with_trace(payload, "ok", len(all_pmids), len(pairs))
