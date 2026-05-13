import asyncio
import logging
import os
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

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
try:
    import grpc
except Exception:
    grpc = None
try:
    from google.api_core.exceptions import GoogleAPICallError
except Exception:
    GoogleAPICallError = None
from src.prompts.templates import RPS_EXTRACTOR_SYSTEM_PROMPT, RPS_EXTRACTOR_HUMAN_PROMPT, with_json_system_suffix
from src.pubmed_client import PubMedClient, RawArticle
from src.state.state import GraphState
from src.utils.rps_utils import compute_rps, compute_rps_verbose, grade_from_rps
def score_rps_for_document(doc: dict) -> float:
    """
    Compute the RPS score for a single document dict using config-driven weights and normalization.
    Returns 0.5 if all features are missing or on error.
    """
    try:
        score = compute_rps(doc)
        # Clamp to [0, 1]
        if not isinstance(score, float):
            return 0.5
        return min(1.0, max(0.0, score))
    except Exception as e:
        logger.warning(f"RPS scoring failed for doc {doc.get('pmid', '-')}: {e}")
        return 0.5


def _attach_rps_metadata(row: Dict[str, Any]) -> Dict[str, Any]:
    details = compute_rps_verbose(row)
    row["rps_score"] = details["rps"]
    row["rps_raw"] = details["rps_raw"]
    row["rps_feature_coverage"] = details["rps_feature_coverage"]
    row["available_features"] = details["available_features"]
    row["missing_features"] = details["missing_features"]
    row["grade"] = grade_from_rps(row["rps_score"])
    return row
from src.utils.epistemic_trace import build_trace_event, build_trace_updates

logger = logging.getLogger(__name__)

MAX_PAPERS = 15
BATCH_SIZE = 1
from src.config import settings

# Force a single concurrent RPS batch to avoid VRAM/OOM crashes on small GPUs.
# Running multiple heavy scoring windows concurrently can exhaust a 4GB VRAM device.
MAX_CONCURRENT_RPS_BATCHES = 1

SKIP_SAMPLE_SIZE_STUDY_TYPES = {
    "expert opinion",
    "editorial",
    "guideline",
    "practice guideline",
    "review",
}


def _is_skip_study_type(study_type: str) -> bool:
    st = str(study_type or "").strip().lower()
    return st in SKIP_SAMPLE_SIZE_STUDY_TYPES


def _pick_study_type(article: RawArticle) -> str:
    study_types = list(article.study_types or article.publication_types or [])
    if not study_types:
        abstract_text = f"{article.title} {article.abstract}"
        keyword_map = [
            (r"meta[- ]analy", "Meta-Analysis"),
            (r"systematic review", "Systematic Review"),
            (r"randomized|randomised|\brct\b", "Randomized Controlled Trial"),
            (r"prospective cohort|retrospective cohort|cohort study", "Cohort"),
            (r"case report", "Case Reports"),
            (r"case series", "Case Series"),
            (r"case[- ]control", "Case-Control"),
            (r"cross[- ]sectional", "Cross-Sectional"),
            (r"review", "Review"),
        ]
        for pattern, mapped in keyword_map:
            if re.search(pattern, abstract_text, flags=re.IGNORECASE):
                return mapped
        return "Unknown"
    for st in study_types:
        if _is_skip_study_type(st):
            return st
    return str(study_types[0])

_CUSTOM_CACHE_DIR = os.getenv("MRAGE_RPS_CACHE_DIR")
if _CUSTOM_CACHE_DIR:
    _RPS_CACHE_PATH = Path(_CUSTOM_CACHE_DIR) / "rps_abstract_cache.sqlite3"
else:
    _RPS_CACHE_PATH = Path(__file__).resolve().parents[2] / "data" / "rps_abstract_cache.sqlite3"
logger.info("RPS cache path: %s", _RPS_CACHE_PATH)

_RPS_CACHE_TABLE = "rps_abstract_cache"
_RPS_CACHE_AVAILABLE = True


def _init_rps_cache() -> None:
    global _RPS_CACHE_AVAILABLE
    if not _RPS_CACHE_AVAILABLE:
        return
    try:
        _RPS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(_RPS_CACHE_PATH) as conn:
            conn.execute(
                f"CREATE TABLE IF NOT EXISTS {_RPS_CACHE_TABLE} ("
                "pmid TEXT PRIMARY KEY, "
                "abstract TEXT NOT NULL, "
                "updated_at TEXT"
                ")"
            )
            conn.commit()
    except Exception as exc:
        logger.warning("RPS cache init failed; disabling cache: %s", exc)
        _RPS_CACHE_AVAILABLE = False


def _load_cached_abstracts(pmids: List[str]) -> Dict[str, str]:
    if not pmids or not _RPS_CACHE_AVAILABLE:
        return {}
    try:
        _init_rps_cache()
        placeholders = ",".join(["?"] * len(pmids))
        query = f"SELECT pmid, abstract FROM {_RPS_CACHE_TABLE} WHERE pmid IN ({placeholders})"
        with sqlite3.connect(_RPS_CACHE_PATH) as conn:
            rows = conn.execute(query, pmids).fetchall()
        return {str(row[0]): str(row[1]) for row in rows}
    except Exception as exc:
        logger.warning("RPS cache read failed; skipping cache: %s", exc)
        return {}


def _store_cached_abstracts(abstracts: Dict[str, str]) -> None:
    if not abstracts or not _RPS_CACHE_AVAILABLE:
        return
    try:
        _init_rps_cache()
        now = datetime.utcnow().isoformat()
        rows = [(pmid, abstract, now) for pmid, abstract in abstracts.items()]
        with sqlite3.connect(_RPS_CACHE_PATH) as conn:
            conn.executemany(
                f"INSERT OR REPLACE INTO {_RPS_CACHE_TABLE} (pmid, abstract, updated_at) "
                "VALUES (?, ?, ?)",
                rows,
            )
            conn.commit()
    except Exception as exc:
        logger.warning("RPS cache write failed; continuing without cache: %s", exc)
    

def _trim_abstract_preserve_results(abstract: str, max_len: int = 1500) -> str:
    """Trim the abstract while preserving Methods/Results sections when possible.

    Strategy:
    - If a 'Results' (or 'Conclusions') marker exists, preserve it and any
      preceding 'Methods' section. If that preserved block fits under max_len,
      prepend a bit of leading context to reach max_len.
    - Otherwise, keep the last `max_len` characters (Results are often at the end).
    """
    if not abstract:
        return ""
    a = str(abstract).strip()
    lower = a.lower()

    results_markers = ["results:", "results.", "conclusions:", "conclusion:", "findings:", "outcomes:"]
    methods_markers = ["methods:", "materials and methods:", "patients and methods:", "method:", "study design:"]

    results_pos = None
    for marker in results_markers:
        idx = lower.find(marker)
        if idx != -1:
            results_pos = idx
            break

    methods_pos = None
    for marker in methods_markers:
        idx = lower.find(marker)
        if idx != -1:
            methods_pos = idx
            break

    if results_pos is not None:
        start = methods_pos if (methods_pos is not None and methods_pos < results_pos) else results_pos
        preserved = a[start:]
        if len(preserved) <= max_len:
            lead_len = max(0, max_len - len(preserved))
            lead = a[:lead_len]
            combined = (lead + "\n\n" + preserved)[:max_len]
            return combined
        return preserved[:max_len]

    # Fallback: keep last max_len chars to preserve Results often placed at the end
    if len(a) <= max_len:
        return a
    return a[-max_len:]


class RPSScoredNote(BaseModel):
    fact: str = Field(default="")
    study_type: str = Field(default="Unspecified")
    sample_size: Any = Field(default=None)
    effect_size: Any = Field(default=None)
    p_value_reported: bool = Field(default=False)
    pre_registered: bool = Field(default=False)
    multi_center: bool = Field(default=False)
    industry_funded: Optional[bool] = Field(default=None)
    confidence: float = Field(default=0.0)


class RPSScoredNotesOutput(BaseModel):
    scored_notes: List[RPSScoredNote] = Field(default_factory=list)


def _build_rps_chain():
    llm = ModelRegistry.get_flash_llm(temperature=0.0, json_mode=True)
    if llm is None:
        raise RuntimeError("rps_scoring_node: Flash LLM unavailable")
    parser = JsonOutputParser(pydantic_object=RPSScoredNotesOutput)
    prompt = ChatPromptTemplate.from_messages([
        ("system", with_json_system_suffix(RPS_EXTRACTOR_SYSTEM_PROMPT)),
        ("human", RPS_EXTRACTOR_HUMAN_PROMPT),
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


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential_jitter(initial=2, max=10),
    retry=retry_if_exception_type((asyncio.TimeoutError, Exception)),
)
async def _score_abstract_batch(
    batch: List[Dict[str, Any]],
    chain,
    step_index_by_pmid: Dict[str, int] | None = None,
) -> List[Dict[str, Any]]:
    """Score a batch of abstracts via LLM extraction.

    Args:
        batch: list of dicts with keys pmid, step_idx, abstract.
        chain: the LangChain chain to invoke for extraction.
        step_index_by_pmid: mapping from PMID string to step index.
            Previously this was a closure variable from rps_scoring_node(),
            which caused a NameError at runtime since this function is
            defined at module scope.
    """
    if step_index_by_pmid is None:
        step_index_by_pmid = {}
    study_type_by_pmid = {
        str(item.get("pmid")): item.get("study_type")
        for item in batch
        if item.get("pmid") is not None
    }
    notes_with_context = "\n\n".join([
        f"Fact {idx + 1}: PMID {item['pmid']} abstract\n{item['abstract']}"
        for idx, item in enumerate(batch)
    ])

    try:
        # Give heavier RPS LLM calls more runway so Tenacity retries can succeed
        result = await safe_ainvoke(chain, {"notes_with_context": notes_with_context}, timeout=300.0)
        scored_notes = result.get("scored_notes", [])
        output: List[Dict[str, Any]] = []

        for idx, item in enumerate(scored_notes):
            # Try to extract PMID from the LLM-provided `fact` text. The
            # prompt prefixes each input with "Fact N: PMID <pmid> ..." so
            # compliant outputs may include that PMID. If present, use it to
            # map the scored note back to the correct article. Otherwise
            # fall back to positional mapping into the input `batch`.
            fact_text = str(item.get("fact", "") or "").strip()
            pmid_from_fact = None
            m = re.search(r"\bPMID\b[:#]?\s*(\d{4,10})", fact_text, flags=re.IGNORECASE)
            if m:
                pmid_from_fact = m.group(1)

            if pmid_from_fact:
                fallback_pmid = pmid_from_fact
                fallback_step_idx = step_index_by_pmid.get(fallback_pmid, -1)
            else:
                if idx < len(batch):
                    fallback_pmid = batch[idx].get("pmid", "unknown")
                    fallback_step_idx = batch[idx].get("step_idx", -1)
                else:
                    # LLM returned more scored_notes than input batch items
                    # and no PMID was extractable from the fact text. Skip
                    # these excess items — they have no provenance and would
                    # produce 'PMID unknown' rows that degrade RPS metrics.
                    logger.debug(
                        "RPS batch: skipping excess scored_note idx=%d "
                        "(batch has %d items, no PMID in fact text)",
                        idx, len(batch),
                    )
                    continue

            sample_raw = str(item.get("sample_size", ""))
            sample_digits = re.sub(r"[^0-9]", "", sample_raw)
            try:
                sample_size = int(sample_digits) if sample_digits else None
            except ValueError:
                sample_size = None

            effect_raw = str(item.get("effect_size", ""))
            effect_clean = re.sub(r"[^0-9eE+\-.]", "", effect_raw)
            try:
                effect_size = float(effect_clean) if effect_clean else None
            except ValueError:
                effect_size = None

            study_type_hint = study_type_by_pmid.get(str(fallback_pmid))
            row = {
                "pmid": fallback_pmid,
                "target_pmid": fallback_pmid,
                "step_idx": fallback_step_idx,
                "fact": item.get("fact", ""),
                "study_type": item.get("study_type") or study_type_hint or "Unspecified",
                "sample_size": sample_size,
                "effect_size": effect_size,
                "p_value_reported": bool(item.get("p_value_reported", False)),
                "pre_registered": bool(item.get("pre_registered", False)),
                "multi_center": bool(item.get("multi_center", False)),
                "industry_funded": item.get("industry_funded"),
                "confidence": float(item.get("confidence", 0.0)),
            }
            # Pass through the abstract text so compute_rps can use
            # get_sample_size_from_abstract() as a secondary fallback.
            abstract_text = ""
            for b in batch:
                if str(b.get("pmid")) == str(fallback_pmid):
                    abstract_text = b.get("abstract", "")
                    break
            row["abstract"] = abstract_text
            output.append(_attach_rps_metadata(row))

        # If parser returns fewer rows than inputs, fill conservatively.
        # Include abstract + call compute_rps so study-type proxy kicks in.
        while len(output) < len(batch):
            fallback = batch[len(output)]
            fallback_row = {
                "pmid": fallback["pmid"],
                "target_pmid": fallback["pmid"],
                "step_idx": fallback["step_idx"],
                "fact": "",
                "study_type": fallback.get("study_type", "Unknown"),
                "sample_size": None,
                "effect_size": None,
                "p_value_reported": False,
                "pre_registered": False,
                "multi_center": False,
                "industry_funded": None,
                "confidence": 0.0,
                "abstract": fallback.get("abstract", ""),
            }
            output.append(_attach_rps_metadata(fallback_row))

        return output

    except Exception as exc:
        # If this is a transient timeout or API-level error, re-raise so the
        # tenacity retry wrapper can retry. Otherwise, fall back to neutral
        # scores to keep the pipeline moving.
        is_transient = False
        if isinstance(exc, asyncio.TimeoutError):
            is_transient = True
        if grpc is not None and isinstance(exc, grpc.RpcError):
            is_transient = True
        if GoogleAPICallError is not None and isinstance(exc, GoogleAPICallError):
            is_transient = True
        if is_transient:
            logger.warning("RPS batch scoring transient error, will retry: %s", exc, exc_info=True)
            raise
        logger.warning("RPS batch scoring failed for %s abstracts: %s", len(batch), exc)
        # Return scores computed via study-type proxy rather than None
        # to avoid downstream NoneType crashes in decision_alignment.
        fallback_rows = []
        for item in batch:
            row = {
                "pmid": item["pmid"],
                "target_pmid": item["pmid"],
                "step_idx": item["step_idx"],
                "fact": "",
                "study_type": item.get("study_type", "Unknown"),
                "sample_size": None,
                "effect_size": None,
                "p_value_reported": False,
                "pre_registered": False,
                "multi_center": False,
                "industry_funded": None,
                "confidence": 0.0,
                "abstract": item.get("abstract", ""),
            }
            fallback_rows.append(_attach_rps_metadata(row))
        return fallback_rows


async def rps_scoring_node(state: GraphState) -> Dict[str, Any]:
    """
    Reads:  state["step_docs_ids"]
    Writes: state["rps_scores"]
    """
    step_doc_ids = state.get("step_docs_ids", [])
    pmids = _flatten_unique_pmids(step_doc_ids)[:MAX_PAPERS]
    if not pmids:
        logger.info("rps_scoring_node: no PMIDs to score, returning neutral placeholder")
        # Return a neutral placeholder so downstream nodes treat RPS as unavailable
        # but not as an extreme low value. Downstream consumers should interpret
        # pmid=None / step_idx=-1 as a placeholder entry.
        neutral_details = compute_rps_verbose({})
        payload = {
            "rps_scores": [
                {
                    "pmid": None,
                    "step_idx": -1,
                    "fact": "",
                    "study_type": "Unknown",
                    "sample_size": None,
                    "effect_size": None,
                    "p_value_reported": False,
                    "pre_registered": False,
                    "multi_center": False,
                    "industry_funded": None,
                    "confidence": 0.0,
                    "rps_score": 0.5,
                    "rps_raw": neutral_details["rps_raw"],
                    "rps_feature_coverage": neutral_details["rps_feature_coverage"],
                    "available_features": neutral_details["available_features"],
                    "missing_features": neutral_details["missing_features"],
                    "grade": "C",
                }
            ]
        }
        trace_event = build_trace_event(
            state,
            section="evidence_analysis",
            event="rps_scoring",
            node="rps_scoring",
            data={
                "pmid_count": 0,
                "scored_count": 0,
                "avg_rps": 0.5,
                "note": "no_pmids",
            },
            influence={"state_updates": ["rps_scores"]},
            attach_context=False,
        )
        trace_updates = build_trace_updates(state, [trace_event])
        payload["trace_events"] = payload.get("trace_events", []) + trace_updates.get("trace_events", [])
        if "trace_id" not in payload:
            payload["trace_id"] = trace_updates.get("trace_id")
        if "trace_created_at" not in payload:
            payload["trace_created_at"] = trace_updates.get("trace_created_at")
        return payload

    client = PubMedClient(
        filter_humans=False,
        filter_recent_years=None,
        filter_study_types=False,
    )
    cached_abstracts = _load_cached_abstracts(pmids)
    missing_pmids = [pmid for pmid in pmids if pmid not in cached_abstracts]
    if cached_abstracts:
        logger.info("RPS cache hit: %d/%d abstracts", len(cached_abstracts), len(pmids))

    raw_articles: List[RawArticle] = []
    for pmid in pmids:
        abstract = cached_abstracts.get(pmid)
        if abstract:
            raw_articles.append(RawArticle(pmid=pmid, abstract=abstract))

    if missing_pmids:
        fetched_articles: List[RawArticle] = await client.efetch(missing_pmids)
        raw_articles.extend(fetched_articles)
        new_cache = {
            str(article.pmid): str(article.abstract)
            for article in fetched_articles
            if article.pmid and article.abstract
        }
        _store_cached_abstracts(new_cache)

    step_index_by_pmid: Dict[str, int] = {}
    for idx, ids in enumerate(step_doc_ids):
        if isinstance(ids, list):
            for pmid in ids:
                value = str(pmid)
                if value and value not in step_index_by_pmid:
                    step_index_by_pmid[value] = idx

    paper_rows: List[Dict[str, Any]] = []
    for article in raw_articles:
        if not article.abstract or not article.pmid:
            continue
        pmid = str(article.pmid)
        if pmid in pmids:
            study_type = _pick_study_type(article)
            paper_rows.append(
                {
                    "pmid": pmid,
                    "step_idx": step_index_by_pmid.get(pmid, -1),
                    "study_type": study_type,
                    # Smart trim to reduce prompt size while preserving Methods/Results
                    "abstract": _trim_abstract_preserve_results(str(article.abstract), max_len=1500),
                }
            )

    if not paper_rows:
        return {"rps_scores": []}

    chain = _build_rps_chain()
    batches = [paper_rows[i: i + BATCH_SIZE] for i in range(0, len(paper_rows), BATCH_SIZE)]
    sem = asyncio.Semaphore(MAX_CONCURRENT_RPS_BATCHES)

    def _build_skip_rows(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for item in items:
            row = {
                "pmid": item["pmid"],
                "target_pmid": item["pmid"],
                "step_idx": item["step_idx"],
                "fact": "",
                "study_type": item.get("study_type", "Unknown"),
                "sample_size": 0,
                "effect_size": None,
                "p_value_reported": False,
                "pre_registered": False,
                "multi_center": False,
                "industry_funded": None,
                "confidence": 0.0,
                "abstract": item.get("abstract", ""),
                "skip_sample_size_extraction": True,
            }
            rows.append(_attach_rps_metadata(row))
        return rows

    async def _bounded_score(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        async with sem:
            skip_items: List[Dict[str, Any]] = []
            llm_items: List[Dict[str, Any]] = []
            for item in batch:
                if _is_skip_study_type(item.get("study_type", "")):
                    skip_items.append(item)
                else:
                    llm_items.append(item)
            rows: List[Dict[str, Any]] = []
            if skip_items:
                rows.extend(_build_skip_rows(skip_items))
            if llm_items:
                rows.extend(await _score_abstract_batch(llm_items, chain, step_index_by_pmid))
            return rows

    batch_results = await asyncio.gather(
        *[_bounded_score(batch) for batch in batches],
        return_exceptions=True,
    )

    rps_scores: List[Dict[str, Any]] = []
    for idx, result in enumerate(batch_results):
        if isinstance(result, Exception):
            logger.warning("RPS batch %s failed unexpectedly: %s", idx + 1, result)
            fallback_batch = batches[idx]
            for item in fallback_batch:
                skip_sample = _is_skip_study_type(item.get("study_type", ""))
                row = {
                    "pmid": item["pmid"],
                    "target_pmid": item["pmid"],
                    "step_idx": item["step_idx"],
                    "fact": "",
                    "study_type": item.get("study_type", "Unknown"),
                    "sample_size": 0 if skip_sample else None,
                    "effect_size": None,
                    "p_value_reported": False,
                    "pre_registered": False,
                    "multi_center": False,
                    "industry_funded": None,
                    "confidence": 0.0,
                    "abstract": item.get("abstract", ""),
                    "skip_sample_size_extraction": skip_sample,
                }
                rps_scores.append(_attach_rps_metadata(row))
        else:
            rps_scores.extend(result)

    # Defensive: filter out None/unparseable values before averaging
    valid_scores: List[float] = []
    for s in rps_scores:
        val = s.get("final_score", s.get("rps_score"))
        if val is None:
            continue
        try:
            valid_scores.append(float(val))
        except (TypeError, ValueError):
            continue
    avg_rps = sum(valid_scores) / len(valid_scores) if valid_scores else None
    logger.info(
        f"rps_scoring_node complete: {len(rps_scores)} entries, "
        f"avg_rps={avg_rps}"
    )
    sample_missing = 0
    for s in rps_scores:
        if s.get("sample_size") in (None, 0, "", "unknown"):
            sample_missing += 1

    payload = {"rps_scores": rps_scores}
    trace_event = build_trace_event(
        state,
        section="evidence_analysis",
        event="rps_scoring",
        node="rps_scoring",
        data={
            "pmid_count": len(pmids),
            "scored_count": len(rps_scores),
            "avg_rps": round(avg_rps, 3) if avg_rps is not None else None,
            "min_rps": round(min(valid_scores), 3) if valid_scores else None,
            "max_rps": round(max(valid_scores), 3) if valid_scores else None,
            "sample_size_missing": sample_missing,
        },
        influence={"state_updates": ["rps_scores"]},
        attach_context=False,
    )
    trace_updates = build_trace_updates(state, [trace_event])
    payload["trace_events"] = payload.get("trace_events", []) + trace_updates.get("trace_events", [])
    if "trace_id" not in payload:
        payload["trace_id"] = trace_updates.get("trace_id")
    if "trace_created_at" not in payload:
        payload["trace_created_at"] = trace_updates.get("trace_created_at")
    return payload
