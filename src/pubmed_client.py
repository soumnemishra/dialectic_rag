# FILE: src/pubmed_client.py
"""
PubMed E-utilities client with 2-stage retrieval pipeline.

Implements a production-grade retrieval system:
    Doctor Query → Structured Query → eSearch → eFetch → Filtering → Pydantic Objects

Architecture:
    1. eSearch: Query → PMIDs (JSON response)
    2. eFetch: PMIDs → Full articles (XML response)
    3. Post-retrieval filtering (Humans, date, study type)
    4. Pydantic validation layer

Example Usage:
    from src.pubmed_client import PubMedClient
    
    client = PubMedClient()
    results = await client.search("diabetes treatment")
    for doc in results:
        print(doc.pmid, doc.title)
"""

import logging
import asyncio
import json
import os
import re
from collections import deque
import aiohttp
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, List, Optional, Set

from bs4 import BeautifulSoup
from pydantic import BaseModel, Field, field_validator
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential_jitter,
    wait_exponential,
    retry_if_exception_type,
)

from src.config import settings
from src.exceptions import PubMedAPIError, TransientError
from src.core.registry import ModelRegistry
from src.utils.debug_utils import get_debug_manager

logger = logging.getLogger(__name__)


class _SlidingWindowRateLimiter:
    """Shared sliding-window limiter for outbound NCBI requests."""

    def __init__(self, max_calls: int, period_seconds: float = 1.0):
        self.max_calls = max_calls
        self.period_seconds = period_seconds
        self._timestamps = deque()
        self._lock = asyncio.Lock()

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def acquire(self) -> None:
        async with self._lock:
            now = asyncio.get_running_loop().time()
            while self._timestamps and now - self._timestamps[0] >= self.period_seconds:
                self._timestamps.popleft()

            if len(self._timestamps) >= self.max_calls:
                sleep_time = self.period_seconds - (now - self._timestamps[0])
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                now = asyncio.get_running_loop().time()
                while self._timestamps and now - self._timestamps[0] >= self.period_seconds:
                    self._timestamps.popleft()

            self._timestamps.append(now)


_RATE_LIMITERS: dict[int, _SlidingWindowRateLimiter] = {}


def _get_rate_limiter(requests_per_second: int) -> _SlidingWindowRateLimiter:
    limiter = _RATE_LIMITERS.get(requests_per_second)
    if limiter is None:
        limiter = _SlidingWindowRateLimiter(requests_per_second, 1.0)
        _RATE_LIMITERS[requests_per_second] = limiter
    return limiter


# =============================================================================
# Enums and Constants
# =============================================================================

class StudyType(str, Enum):
    """Accepted study types for clinical relevance."""
    CLINICAL_TRIAL = "Clinical Trial"
    RCT = "Randomized Controlled Trial"
    SYSTEMATIC_REVIEW = "Systematic Review"
    META_ANALYSIS = "Meta-Analysis"
    REVIEW = "Review"
    CASE_REPORT = "Case Reports"
    GUIDELINE = "Practice Guideline"
    OBSERVATIONAL = "Observational Study"


ALLOWED_STUDY_TYPES: Set[str] = {st.value for st in StudyType}


# =============================================================================
# Pydantic Models (Validation Layer)
# =============================================================================

class RetrievedDocument(BaseModel):
    """
    Validated PubMed document ready for RAG pipeline.
    
    This is the FINAL output - clean, validated, and safe.
    
    Attributes:
        pmid: PubMed identifier (unique).
        title: Article title.
        abstract: Full abstract text.
        year: Publication year.
        authors: List of author names.
        journal: Journal name.
        doi: Digital Object Identifier (optional).
        study_types: Publication types.
        mesh_terms: MeSH descriptor terms.
        source: Always "PubMed".
    """
    pmid: str = Field(description="PubMed ID")
    title: str = Field(description="Article title")
    abstract: str = Field(min_length=1, description="Abstract text")
    year: Optional[int] = Field(default=None, description="Publication year")
    authors: List[str] = Field(default_factory=list, description="Author names")
    journal: str = Field(default="", description="Journal name")
    doi: Optional[str] = Field(default=None, description="DOI")
    study_types: List[str] = Field(default_factory=list, description="Publication types")
    mesh_terms: List[str] = Field(default_factory=list, description="MeSH terms")
    is_retracted: bool = Field(default=False, description="Whether PubMed marks the article with a retraction/correction relationship")
    source: str = Field(default="PubMed", description="Data source")
    publication_types: List[str] = Field(default_factory=list, description="All publication types found")
    # Provenance and retrieval metadata (populated by retrieval layer)
    original_pmid: Optional[str] = Field(default=None, description="Original PubMed PMID if present")
    retrieval_method: Optional[str] = Field(default=None, description="Method used to retrieve this record (e.g., efetch)")
    original_rank: Optional[int] = Field(default=None, description="Original rank/order returned by PubMed eSearch/efetch")
    retrieval_score: Optional[float] = Field(default=None, description="Optional retrieval score from reranker/dense retriever")
    canonical_id: Optional[str] = Field(default=None, description="Canonical identifier (PMID or DOI or title hash)")
    
    @field_validator("abstract")
    @classmethod
    def abstract_not_empty(cls, v: str) -> str:
        """Reject articles without abstracts."""
        if not v or not v.strip():
            raise ValueError("Abstract cannot be empty")
        return v.strip()
    
    def to_context_string(self) -> str:
        """Format document as context string for LLM."""
        authors_str = ", ".join(self.authors[:3])
        if len(self.authors) > 3:
            authors_str += " et al."
        
        study_str = ", ".join(self.study_types[:2]) if self.study_types else "Article"
        
        return (
            f"**{self.title}**\n"
            f"Authors: {authors_str}\n"
            f"Journal: {self.journal} ({self.year or 'N/A'})\n"
            f"Type: {study_str}\n"
            f"PMID: {self.pmid}\n\n"
            f"{self.abstract}\n"
        )

    def to_citation(self) -> str:
        """Format document as a citation string."""
        return f"{self.authors[0] if self.authors else 'Unknown'} et al., '{self.title}' (PMID: {self.pmid})"


class SearchMetadata(BaseModel):
    """Metadata from PubMed search."""
    total_count: int = Field(description="Total results in PubMed")
    returned_count: int = Field(description="PMIDs returned")
    query_translation: str = Field(default="", description="PubMed's query translation")


# =============================================================================
# Raw Article (Pre-validation)
# =============================================================================

@dataclass
class RawArticle:
    """
    Raw parsed article before validation.

    This is an intermediate format between XML parsing and Pydantic validation.
    """
    pmid: Optional[str] = None
    title: str = ""
    abstract: str = ""
    year: Optional[int] = None
    authors: List[str] = field(default_factory=list)
    journal: str = ""
    doi: Optional[str] = None
    study_types: List[str] = field(default_factory=list)
    mesh_terms: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    affiliations: List[str] = field(default_factory=list)
    is_human_study: bool = False
    publication_types: List[str] = field(default_factory=list)
    is_retracted: bool = False


# =============================================================================
# PubMed Client
# =============================================================================

class PubMedClient:
    """
    Production-grade PubMed client with 2-stage retrieval.
    
    Pipeline:
        Query → eSearch (PMIDs) → eFetch (XML) → Parse → Filter → Validate
    
    Features:
        - Async I/O with aiohttp
        - Separate search and fetch stages
        - Post-retrieval filtering (humans, date, study type)
        - Pydantic validation layer
        - Exponential backoff retry
        - API key support for higher rate limits
    """
    
    ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        max_results: Optional[int] = None,
        filter_humans: bool = True,
        filter_recent_years: Optional[int] = 7,
        filter_study_types: bool = False,
    ):
        """
        Initialize PubMed client.
        
        Args:
            api_key: NCBI API key for higher rate limits.
            max_results: Maximum PMIDs to retrieve per search.
            filter_humans: Filter for human studies only.
            filter_recent_years: Filter to last N years (None to disable).
            filter_study_types: Filter for clinical study types only.
        """
        self.api_key = api_key or os.getenv("NCBI_API_KEY") or settings.PUBMED_API_KEY
        self.max_results = max_results or settings.MAX_SEARCH_RESULTS
        self.filter_humans = filter_humans
        self.filter_recent_years = filter_recent_years
        self.filter_study_types = filter_study_types
        self.requests_per_second = 10 if self.api_key else 3
        self._rate_limiter = _get_rate_limiter(self.requests_per_second)
        self.esearch_delay_seconds = float(getattr(settings, "PUBMED_ESEARCH_DELAY_SECONDS", 0.0) or 0.0)
        if self.esearch_delay_seconds <= 0:
            self.esearch_delay_seconds = 1.0 / max(self.requests_per_second, 1)
        self._esearch_delay_lock = asyncio.Lock()
        self._esearch_last_call = 0.0
        self.debug_manager = get_debug_manager()
        self.debug_pubmed = os.getenv("DEBUG_PUBMED", "false").strip().lower() in {"1", "true", "yes", "on"}
        self.debug_retrieval = os.getenv("DEBUG_RETRIEVAL", "false").strip().lower() in {"1", "true", "yes", "on"}
        
        logger.info(
            "PubMed client initialized (Async)",
            extra={
                "max_results": self.max_results,
                "filter_humans": filter_humans,
                "filter_recent_years": filter_recent_years,
            }
        )

    def _debug_enabled(self, flag: bool) -> bool:
        return bool(flag and self.debug_manager.is_enabled())

    def _article_debug_payload(self, article: RawArticle) -> dict[str, Any]:
        return {
            "pmid": article.pmid,
            "title": article.title,
            "abstract": article.abstract,
            "year": article.year,
            "journal": article.journal,
            "authors": article.authors,
            "doi": article.doi,
            "publication_types": article.publication_types,
            "mesh_terms": article.mesh_terms,
            "keywords": article.keywords,
            "affiliations": article.affiliations,
        }
    
    def _get_base_params(self) -> dict:
        """Get base parameters including API key."""
        params = {"db": "pubmed"}
        if self.api_key:
            params["api_key"] = self.api_key
        return params

    def _retry_after_seconds(self, header: Optional[str]) -> Optional[float]:
        if not header:
            return None
        value = header.strip()
        if not value:
            return None
        try:
            return max(0.0, float(value))
        except ValueError:
            return None

    async def _handle_rate_limit(self, stage: str, response: aiohttp.ClientResponse) -> None:
        retry_after = self._retry_after_seconds(response.headers.get("Retry-After"))
        if retry_after:
            await asyncio.sleep(retry_after)
        raise TransientError(f"{stage} rate limited (HTTP 429)")

    async def _respect_esearch_delay(self) -> None:
        if self.esearch_delay_seconds <= 0:
            return
        async with self._esearch_delay_lock:
            loop = asyncio.get_running_loop()
            now = loop.time()
            elapsed = now - self._esearch_last_call
            if elapsed < self.esearch_delay_seconds:
                await asyncio.sleep(self.esearch_delay_seconds - elapsed)
            self._esearch_last_call = loop.time()

    @retry(
        stop=stop_after_attempt(settings.RETRY_MAX_ATTEMPTS),
        wait=wait_exponential_jitter(initial=settings.RETRY_WAIT_SECONDS, max=10),
        retry=retry_if_exception_type(TransientError),
        reraise=True,
    )
    async def get_count(self, query: str) -> int:
        """Return the number of results for a given PubMed query."""
        try:
            # Reuse esearch implementation in count-only mode so tests and
            # instrumentation observe the same parameter usage (retmax=0).
            _, metadata = await self.esearch(query, max_results=0)
            logger.info("eSearch count: %d", metadata.total_count)
            return metadata.total_count
        except PubMedAPIError:
            return None
        except Exception as exc:
            logger.warning("get_count failed: %s", exc)
            return None
    
    # =========================================================================
    # STAGE 1: eSearch - Query → PMIDs
    # =========================================================================
    
    @retry(
        stop=stop_after_attempt(settings.RETRY_MAX_ATTEMPTS),
        wait=wait_exponential_jitter(initial=settings.RETRY_WAIT_SECONDS, max=10),
        retry=retry_if_exception_type(TransientError),
        reraise=True,
    )
    async def esearch(self, query: str, max_results: Optional[int] = None) -> tuple[List[str], SearchMetadata]:
        """
        Stage 1: Search PubMed and get PMIDs.
        
        Args:
            query: PubMed query string.
            max_results: Override default max results.
        
        Returns:
            Tuple of (list of PMIDs, search metadata).
        
        Raises:
            PubMedAPIError: On API failure.
        """
        params = self._get_base_params()
        # Respect explicit max_results including 0 (count-only), otherwise use default
        retmax_val = self.max_results if max_results is None else max_results
        params.update({
            "term": query,
            "retmax": str(retmax_val),
            "retmode": "json",
            "sort": "relevance",
        })
        
        logger.info(f"eSearch: {query[:200]}", extra={"query": query[:100]})
        
        try:

            # Create a new session for each attempt to avoid closed loop issues
            async with aiohttp.ClientSession() as session:
                async with self._rate_limiter:
                    await self._respect_esearch_delay()
                    async with session.get(
                                self.ESEARCH_URL,
                                params=params,
                                timeout=aiohttp.ClientTimeout(total=25.0),
                            ) as response:
                                if response.status == 429:
                                    logger.warning("eSearch rate limited", extra={"retry_after": response.headers.get("Retry-After")})
                                    await self._handle_rate_limit("eSearch", response)
                                response.raise_for_status()
                                content_type = response.headers.get("Content-Type", "")

                                # 1) Prefer the JSON helper (works with real aiohttp responses and AsyncMock in tests)
                                data = None
                                try:
                                    data = await response.json()
                                except Exception:
                                    # 2) Fallback: fetch text then parse JSON
                                    try:
                                        raw_text = await response.text(errors="replace")
                                    except TypeError:
                                        # Some mocks may not accept 'errors' kwarg
                                        raw_text = await response.text()

                                    try:
                                        data = json.loads(raw_text, strict=False)
                                    except json.JSONDecodeError as exc:
                                        # Attempt to clean control chars and retry
                                        cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", raw_text)
                                        try:
                                            data = json.loads(cleaned, strict=False)
                                        except json.JSONDecodeError as exc2:
                                            snippet = (cleaned[:200] if isinstance(cleaned, str) else str(cleaned))
                                            raise PubMedAPIError(
                                                "Failed to parse eSearch response",
                                                {
                                                    "error": str(exc2),
                                                    "content_type": content_type,
                                                    "snippet": snippet,
                                                },
                                            ) from exc2

                                # Ensure we have a mapping / dict-like result
                                if not isinstance(data, dict):
                                    # If it's a bytes/str payload, try decode/loads once more
                                    if isinstance(data, (bytes, bytearray)):
                                        try:
                                            data = json.loads(data.decode("utf-8", errors="replace"), strict=False)
                                        except Exception as exc:
                                            raise PubMedAPIError("eSearch returned non-dict payload", {"type": type(data).__name__, "error": str(exc)}) from exc
                                    else:
                                        raise PubMedAPIError("eSearch returned unexpected payload type", {"type": type(data).__name__})
            
            result = data.get("esearchresult", {})
            pmids = result.get("idlist", [])
            metadata = SearchMetadata(
                total_count=int(result.get("count", 0)),
                returned_count=len(pmids),
                query_translation=result.get("querytranslation", ""),
            )
            
            logger.info(
                "eSearch complete",
                extra={
                    "pmid_count": len(pmids),
                    "total_available": metadata.total_count,
                }
            )
            
            return pmids, metadata
            
        except asyncio.TimeoutError as e:
            # Surface timeout as PubMedAPIError for callers/tests
            raise PubMedAPIError("eSearch timeout") from e
        except aiohttp.ClientError as e:
            raise TransientError(f"eSearch failed: {e}") from e
        except (KeyError, ValueError) as e:
            raise PubMedAPIError("Failed to parse eSearch response", {"error": str(e)}) from e
    
    # =========================================================================
    # STAGE 2: eFetch - PMIDs → Articles
    # =========================================================================
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((TransientError, asyncio.TimeoutError)),
        reraise=True,
    )
    async def efetch(self, pmids: List[str]) -> List[RawArticle]:
        """
        Stage 2: Fetch full article data for PMIDs.
        
        Args:
            pmids: List of PubMed IDs.
        
        Returns:
            List of RawArticle objects (pre-validation).
        
        Raises:
            PubMedAPIError: On API failure.
        """
        if not pmids:
            return []
        
        params = self._get_base_params()
        params.update({
            "id": ",".join(pmids),
            "retmode": "xml",
        })
        
        logger.info("eFetch", extra={"pmid_count": len(pmids)})
        
        try:
            async with aiohttp.ClientSession() as session:
                async with self._rate_limiter:
                    async with session.get(
                        self.EFETCH_URL,
                        params=params,
                        timeout=aiohttp.ClientTimeout(total=45.0),
                    ) as response:
                        if response.status == 429:
                            logger.warning("eFetch rate limited", extra={"retry_after": response.headers.get("Retry-After")})
                            await self._handle_rate_limit("eFetch", response)
                        response.raise_for_status()
                        content = await response.read()

            if self._debug_enabled(self.debug_pubmed):
                xml_text = content.decode("utf-8", errors="replace")
                for pmid in pmids:
                    self.debug_manager.save_xml(f"pubmed/{pmid}_raw.xml", xml_text)

            # Parse XML
            articles = await self._parse_xml(content)

            logger.info("eFetch complete", extra={"articles_parsed": len(articles)})
            return articles

        except asyncio.TimeoutError as e:
            # Allow tenacity to catch and retry on timeouts
            logger.warning("eFetch timed out, raising to trigger retry: %s", str(e))
            if self._debug_enabled(self.debug_pubmed):
                timestamp = datetime.utcnow().isoformat(timespec="seconds").replace(":", "-")
                self.debug_manager.save_exception(f"exceptions/traceback_{timestamp}.txt", e)
            raise
        except aiohttp.ClientError as e:
            # Treat client errors as transient to permit retry attempts
            logger.warning("eFetch client error, raising TransientError to trigger retry: %s", str(e))
            if self._debug_enabled(self.debug_pubmed):
                timestamp = datetime.utcnow().isoformat(timespec="seconds").replace(":", "-")
                self.debug_manager.save_exception(f"exceptions/traceback_{timestamp}.txt", e)
            raise TransientError(f"eFetch failed: {e}") from e
    
    # =========================================================================
    # XML Parsing (Using BeautifulSoup)
    # =========================================================================
    
    async def _parse_xml(self, xml_content: bytes) -> List[RawArticle]:
        """Parse PubMed XML response into RawArticle objects."""
        articles = []
        
        try:
            soup = await asyncio.to_thread(BeautifulSoup, xml_content, "xml")
            
            for article_elem in soup.find_all("PubmedArticle"):
                try:
                    article = self._parse_single_article(article_elem)
                    if article:
                        articles.append(article)
                        if self._debug_enabled(self.debug_pubmed) and article.pmid:
                            self.debug_manager.save_json(
                                f"pubmed/{article.pmid}_parsed.json",
                                self._article_debug_payload(article),
                            )
                except Exception as e:
                    logger.warning(
                        "Failed to parse article",
                        extra={"error": str(e)}
                    )
                    continue
                    
        except Exception as e:
            logger.error("XML parse error", extra={"error": str(e)})
            
        return articles
    
    def _parse_single_article(self, article_elem) -> Optional[RawArticle]:
        """Parse a single PubmedArticle element."""
        # PMID (required)
        pmid_elem = article_elem.find("PMID")
        pmid = pmid_elem.text.strip() if pmid_elem and pmid_elem.text else None
        
        # Title
        title_elem = article_elem.find("ArticleTitle")
        title = title_elem.text.strip() if title_elem and title_elem.text else ""
        
        # Abstract (combine all AbstractText elements)
        abstract_parts = []
        for abs_elem in article_elem.find_all("AbstractText"):
            if abs_elem.text:
                label = abs_elem.get("Label", "")
                if label:
                    abstract_parts.append(f"{label}: {abs_elem.text.strip()}")
                else:
                    abstract_parts.append(abs_elem.text.strip())
        abstract = " ".join(abstract_parts)
        
        # Year - Robust extraction across multiple PubMed date fields
        year = None
        # 1) Try PubDate -> Year
        pub_date = article_elem.find("PubDate")
        if pub_date:
            year_elem = pub_date.find("Year")
            if year_elem and year_elem.text:
                try:
                    year = int(year_elem.text.strip())
                except Exception:
                    year = None
            # Fallback to MedlineDate if Year is missing
            if year is None:
                medline_date = pub_date.find("MedlineDate")
                if medline_date and medline_date.text:
                    import re
                    match = re.search(r'\d{4}', medline_date.text)
                    if match:
                        try:
                            year = int(match.group(0))
                        except Exception:
                            year = None
        # 2) Try ArticleDate (modern PubMed XML)
        if year is None:
            article_date = article_elem.find("ArticleDate")
            if article_date:
                year_elem = article_date.find("Year")
                if year_elem and year_elem.text:
                    try:
                        year = int(year_elem.text.strip())
                    except Exception:
                        year = None
        # 3) Try DateCreated or DateCompleted
        if year is None:
            dc = article_elem.find("DateCreated") or article_elem.find("DateCompleted")
            if dc:
                y = dc.find("Year")
                if y and y.text:
                    try:
                        year = int(y.text.strip())
                    except Exception:
                        year = None

        # Authors
        authors = []
        for author in article_elem.find_all("Author"):
            lastname = author.find("LastName")
            forename = author.find("ForeName")
            if lastname and lastname.text:
                name = lastname.text.strip()
                if forename and forename.text:
                    name = f"{forename.text.strip()} {name}"
                authors.append(name)
        
        # Journal
        journal = ""
        journal_elem = article_elem.find("Journal")
        if journal_elem:
            title_elem = journal_elem.find("Title")
            if title_elem and title_elem.text:
                journal = title_elem.text.strip()
        
        # DOI
        doi = None
        for id_elem in article_elem.find_all("ArticleId"):
            if id_elem.get("IdType") == "doi" and id_elem.text:
                doi = id_elem.text.strip()
                break
        
        # Study Types (Publication Types)
        study_types = []
        pub_types = []
        for pub_type in article_elem.find_all("PublicationType"):
            if pub_type.text:
                pt_text = pub_type.text.strip()
                study_types.append(pt_text)
                pub_types.append(pt_text)
        
        # MeSH Terms
        mesh_terms = []
        for mesh in article_elem.find_all("DescriptorName"):
            if mesh.text:
                mesh_terms.append(mesh.text.strip())

        # Keywords
        keywords = []
        for keyword in article_elem.find_all("Keyword"):
            if keyword.text:
                keywords.append(keyword.text.strip())

        # Affiliations
        affiliations = []
        for aff in article_elem.find_all("Affiliation"):
            if aff.text:
                affiliations.append(aff.text.strip())
        
        # Check if human study
        is_human = "Humans" in mesh_terms

        # Retractions/corrections are explicit PubMed XML relationships. Keep
        # this deterministic so downstream safety gates do not depend on text.
        is_retracted = False
        correction_ref_types = {"RetractionOf", "RetractionIn", "UpdateOf", "ErratumFor"}
        corrections = article_elem.find("CommentsCorrectionsList")
        if corrections:
            for comment in corrections.find_all("CommentsCorrections"):
                ref_type = comment.get("RefType", "")
                if ref_type in correction_ref_types:
                    is_retracted = True
                    break
        
        return RawArticle(
            pmid=pmid,
            title=title,
            abstract=abstract,
            year=year,
            authors=authors,
            journal=journal,
            doi=doi,
            study_types=study_types,
            mesh_terms=mesh_terms,
            keywords=keywords,
            affiliations=affiliations,
            is_human_study=is_human,
            publication_types=pub_types,
            is_retracted=is_retracted
        )
    
    # =========================================================================
    # STAGE 3: Post-Retrieval Filtering
    # =========================================================================
    
    def filter_articles(self, articles: List[RawArticle], original_patient_vignette: Optional[str] = None) -> List[RawArticle]:
        """
        Apply post-retrieval filters.
        """
        filtered = articles
        initial_count = len(articles)
        
        # Filter: Humans only
        if self.filter_humans:
            filtered = [a for a in filtered if a.is_human_study]
            logger.debug(f"After humans filter: {len(filtered)}/{initial_count}")
        
        # Filter: Recent years
        if self.filter_recent_years:
            current_year = datetime.now().year
            cutoff = current_year - self.filter_recent_years
            filtered = [
                a for a in filtered
                if a.year is None or a.year >= cutoff
            ]
            logger.debug(f"After date filter: {len(filtered)}/{initial_count}")
        
        # Filter: Study types
        if self.filter_study_types:
            filtered = [
                a for a in filtered
                if any(st in ALLOWED_STUDY_TYPES for st in a.study_types)
            ]
            logger.debug(f"After study type filter: {len(filtered)}/{initial_count}")

        # Filter: Semantic Relevance (Mandatory Gate)
        if original_patient_vignette and filtered:
            try:
                # Use lightweight embedding model
                model = ModelRegistry.get_sentence_transformer(os.getenv("MRAGE_SEMANTIC_MODEL", "all-MiniLM-L6-v2"))
                if model:
                    import numpy as np
                    from sklearn.metrics.pairwise import cosine_similarity

                    vignette_emb = model.encode([original_patient_vignette])
                    # Encode abstracts (or titles if abstract missing)
                    # SentenceTransformer handles truncation usually to 512, but we prioritize abstract
                    texts = [a.abstract or a.title for a in filtered]
                    text_embs = model.encode(texts)

                    similarities = cosine_similarity(vignette_emb, text_embs)[0]
                    
                    # Threshold logic
                    default_threshold = 0.40
                    if settings.FAST_EPISTEMIC:
                        # In fast mode, we allow more noise to ensure speed/recall balance
                        default_threshold = 0.20
                    
                    threshold = float(os.getenv("MRAGE_SEMANTIC_THRESHOLD", str(default_threshold)))
                    
                    semantic_filtered = []
                    for idx, sim in enumerate(similarities):
                        if sim >= threshold:
                            semantic_filtered.append(filtered[idx])
                        else:
                            logger.info(f"Dropping article {filtered[idx].pmid} due to low semantic similarity: {sim:.3f}")
                    
                    if not semantic_filtered and filtered:
                        logger.warning(
                            f"Semantic relevance gate dropped ALL {len(filtered)} articles. "
                            f"Consider lowering MRAGE_SEMANTIC_THRESHOLD (current={threshold})."
                        )

                    filtered = semantic_filtered
            except Exception as e:
                logger.warning(f"Semantic filtering failed: {e}")
        
        logger.info(
            "Filtering complete",
            extra={"before": initial_count, "after": len(filtered)}
        )
        
        return filtered
    
    # =========================================================================
    # STAGE 4: Pydantic Validation
    # =========================================================================
    
    def validate_articles(self, articles: List[RawArticle]) -> List[RetrievedDocument]:
        """
        Validate articles through Pydantic layer.
        """
        validated = []
        rejected = 0
        
        for idx, article in enumerate(articles):
            try:
                # Create base document
                doc = RetrievedDocument(
                    pmid=article.pmid or "",
                    title=article.title,
                    abstract=article.abstract,
                    year=article.year,
                    authors=article.authors,
                    journal=article.journal,
                    doi=article.doi,
                    study_types=article.study_types,
                    mesh_terms=article.mesh_terms,
                    is_retracted=article.is_retracted,
                    publication_types=article.publication_types,
                )

                # Provenance: record original pmid if available
                doc.original_pmid = article.pmid if article.pmid else None

                # Canonical ID: prefer PMID, then DOI, then title hash
                canonical = None
                if article.pmid:
                    canonical = article.pmid
                elif article.doi:
                    # normalize DOI (lowercase, strip)
                    canonical = f"DOI:{article.doi.strip().lower()}"
                else:
                    # Fallback to normalized title hash
                    import hashlib
                    norm_title = (article.title or "").strip().lower()
                    h = hashlib.sha1(norm_title.encode("utf-8") if isinstance(norm_title, str) else norm_title)
                    canonical = f"TITLEHASH:{h.hexdigest()[:12]}"

                doc.canonical_id = canonical
                # Preserve `pmid` as the original numeric PubMed ID for backward
                # compatibility when available. If the article lacks a PMID,
                # fall back to using the canonical identifier (e.g., DOI:...)
                # as the document `pmid` so downstream consumers continue to
                # see a stable single identifier.
                doc.original_pmid = article.pmid if article.pmid else None
                if article.pmid:
                    doc.pmid = article.pmid
                else:
                    doc.pmid = canonical or ""

                # Retrieval provenance defaults (efetch/parse order)
                doc.retrieval_method = "efetch"
                doc.original_rank = idx
                doc.retrieval_score = None

                validated.append(doc)
            except Exception as e:
                rejected += 1
                logger.debug(
                    "Article rejected by validation",
                    extra={"pmid": article.pmid, "reason": str(e)}
                )
        
        logger.info(
            "Validation complete",
            extra={"validated": len(validated), "rejected": rejected}
        )
        
        return validated
    
    # =========================================================================
    # Main Search Method (Full Pipeline)
    # =========================================================================
    
    async def search(
        self,
        query: str,
        max_results: Optional[int] = None,
        fallback_query: Optional[str] = None,
        min_year: Optional[int] = None,
        max_year: Optional[int] = None,
        original_patient_vignette: Optional[str] = None,
    ) -> List[RetrievedDocument]:
        """
        Execute full 2-stage retrieval pipeline (Async).
        """
        logger.info(f"Starting search pipeline with query: {query[:200]}", extra={"query": query[:100]})
        
        # Stage 1: Search for PMIDs
        pmids, metadata = await self.esearch(query, max_results)
        
        if not pmids:
            logger.info("No PMIDs found", extra={"query": query[:50]})
            return []
        
        # Stage 2: Fetch articles
        try:
            raw_articles = await self.efetch(pmids)
        except Exception as e:
            logger.warning("eFetch failed after retries; returning empty results", extra={"error": str(e)})
            raw_articles = []

        if not raw_articles:
            logger.warning("No articles fetched", extra={"pmid_count": len(pmids)})
            return []
        
        # Stage 3: Filter
        filtered_articles = self.filter_articles(raw_articles, original_patient_vignette=original_patient_vignette)

        # Stage 4: Validate
        validated_docs = self.validate_articles(filtered_articles)

        total_before = len(validated_docs)

        # Optional: Year filtering (caller-specified)
        if min_year is not None or max_year is not None:
            def _year_in_range(d: RetrievedDocument) -> bool:
                if d.year is None:
                    return False
                if min_year is not None and d.year < min_year:
                    return False
                if max_year is not None and d.year > max_year:
                    return False
                return True

            validated_docs = [d for d in validated_docs if _year_in_range(d)]

        # Deduplicate by canonical identifier (prefer `canonical_id`, fallback to `pmid`)
        unique = []
        seen = set()
        for d in validated_docs:
            key = d.canonical_id or d.pmid or ""
            if key in seen:
                continue
            seen.add(key)
            unique.append(d)

        duplicates_removed = total_before - len(unique)

        # Metadata completeness stats
        total = total_before if total_before > 0 else len(unique)
        with_pmid = sum(1 for d in unique if d.original_pmid)
        with_year = sum(1 for d in unique if d.year is not None)
        pct_with_pmid = (with_pmid / total * 100.0) if total else 0.0
        pct_with_year = (with_year / total * 100.0) if total else 0.0

        logger.info(
            "Search metadata completeness",
            extra={
                "query": query[:120],
                "total_validated": total_before,
                "returned": len(unique),
                "percent_with_original_pmid": round(pct_with_pmid, 2),
                "percent_with_year": round(pct_with_year, 2),
                "duplicates_removed": int(duplicates_removed),
            }
        )

        validated_docs = unique
        
        # QUALITY GATE: if no validated docs AND a fallback is provided, try again
        if not validated_docs and fallback_query:
            logger.info("No results for primary query; trying fallback: %s", fallback_query[:100])
            try:
                pmids2, _ = await self.esearch(fallback_query, max_results)
                if pmids2:
                    raw2 = await self.efetch(pmids2)
                    filtered2 = self.filter_articles(raw2, original_patient_vignette=original_patient_vignette)
                    validated_docs = self.validate_articles(filtered2)
            except Exception as e:
                logger.warning("Fallback search failed: %s", e)

        logger.info(
            "Search pipeline complete",
            extra={
                "query": query[:50],
                "pmids": len(pmids),
                "parsed": len(raw_articles),
                "filtered": len(filtered_articles),
                "validated": len(validated_docs),
            }
        )

        if self._debug_enabled(self.debug_retrieval):
            self.debug_manager.save_query_snapshot(
                query=query,
                payload={
                    "pmids": pmids,
                    "retrieved_articles": [d.model_dump() for d in validated_docs],
                    "search_metadata": metadata.model_dump(),
                    "evidence_scores": {},
                    "calibration_metrics": {},
                    "final_answer": None,
                    "epistemic_state": None,
                },
            )
        
        return validated_docs
    
    async def search_with_metadata(
        self,
        query: str,
        max_results: Optional[int] = None,
        min_year: Optional[int] = None,
        max_year: Optional[int] = None,
        original_patient_vignette: Optional[str] = None,
    ) -> tuple[List[RetrievedDocument], SearchMetadata]:
        """
        Search with metadata about the search.
        """
        pmids, metadata = await self.esearch(query, max_results)
        
        if not pmids:
            return [], metadata
        
        # Stage 2 & 3: Fetch and Filter
        raw_articles = await self.efetch(pmids)
        filtered_articles = self.filter_articles(raw_articles, original_patient_vignette=original_patient_vignette)
        
        # Stage 4: Validate
        validated_docs = self.validate_articles(filtered_articles)

        # Apply optional year filtering and dedup similarly to search()
        total_before = len(validated_docs)
        if min_year is not None or max_year is not None:
            def _year_in_range(d: RetrievedDocument) -> bool:
                if d.year is None:
                    return False
                if min_year is not None and d.year < min_year:
                    return False
                if max_year is not None and d.year > max_year:
                    return False
                return True
            validated_docs = [d for d in validated_docs if _year_in_range(d)]

        unique = []
        seen = set()
        for d in validated_docs:
            key = d.canonical_id or d.pmid or ""
            if key in seen:
                continue
            seen.add(key)
            unique.append(d)

        duplicates_removed = total_before - len(unique)
        logger.info(
            "Search_with_metadata completeness",
            extra={
                "query": query[:120],
                "total_validated": total_before,
                "returned": len(unique),
                "duplicates_removed": int(duplicates_removed),
            }
        )

        return unique, metadata


# =============================================================================
# Backward Compatibility Alias
# =============================================================================

# Alias for backward compatibility with existing code
PubMedArticle = RetrievedDocument
