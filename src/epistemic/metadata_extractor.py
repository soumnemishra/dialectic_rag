import logging
import os
import re
from typing import List, Dict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from src.core.registry import ModelRegistry, safe_ainvoke
from src.models.schemas import StudyMetadata, StudyDesign
from src.utils.debug_utils import get_debug_manager

logger = logging.getLogger(__name__)


class MetadataExtractionError(Exception):
    pass


def with_json_system_suffix(system_prompt: str) -> str:
    """Append a strict JSON-only suffix exactly once to a system prompt."""
    JSON_ONLY_SUFFIX = (
        "CRITICAL INSTRUCTION: You must output ONLY valid JSON. Do NOT output any "
        "conversational text, preamble, or markdown formatting before or after the "
        "JSON object. Start your response with an opening curly bracket and end with a closing curly bracket."
    )
    if JSON_ONLY_SUFFIX in system_prompt:
        return system_prompt
    return system_prompt.rstrip() + "\n\n" + JSON_ONLY_SUFFIX


METADATA_EXTRACTION_SYSTEM_PROMPT = """
You are a precise biomedical metadata extractor. 
Extract study features from the provided abstract. 
Return ONLY a JSON object. 
Be honest: if a value is not mentioned, return null. 
Do not hallucinate.

Required keys:
- title: str
- publication_year: int or null
- study_design: RCT | Systematic Review | Meta-Analysis | Cohort | Case-Control | Case Report | other | null
- sample_size: int or null
- p_value_present: bool or null
- ci_present: bool or null
- preregistration_id: str or null (e.g., NCT number)
- limitations: list of str
- journal: str or null
"""

METADATA_EXTRACTION_HUMAN_PROMPT = "Abstract: {abstract}"


class MetadataExtractor:
    def __init__(self):
        self.llm = None
        self.parser = JsonOutputParser()
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", with_json_system_suffix(METADATA_EXTRACTION_SYSTEM_PROMPT)),
            ("human", METADATA_EXTRACTION_HUMAN_PROMPT),
        ])

        self.debug_manager = get_debug_manager()
        self.debug_metadata = os.getenv("DEBUG_METADATA", "false").strip().lower() in {"1", "true", "yes", "on"}

        try:
            self.llm = ModelRegistry.get_flash_llm(temperature=0.0, json_mode=True)
        except Exception as e:
            logger.warning("MetadataExtractor: Flash LLM unavailable; deterministic extraction remains active: %s", e)

    def _debug_enabled(self) -> bool:
        return bool(self.debug_metadata and self.debug_manager.is_enabled())

    def _llm_metadata_payload(self, metadata: StudyMetadata) -> Dict[str, Any]:
        data = metadata.model_dump() if hasattr(metadata, "model_dump") else vars(metadata)
        return {
            "study_design": data.get("study_design"),
            "sample_size": data.get("sample_size"),
            "population": data.get("population"),
            "intervention": data.get("intervention"),
            "comparator": data.get("comparator"),
            "outcomes": data.get("outcomes"),
            "effect_direction": data.get("effect_direction"),
            "confidence": data.get("confidence"),
            "year": data.get("year"),
            "has_p_value": data.get("has_p_value"),
            "has_CI": data.get("has_CI"),
            "preregistration_id": data.get("preregistration_id"),
        }

    def _normalize_design(self, raw: Optional[str]) -> StudyDesign:
        if not raw:
            return StudyDesign.OTHER
        mapping = {
            "meta-analysis": StudyDesign.META_ANALYSIS,
            "meta analysis": StudyDesign.META_ANALYSIS,
            "systematic review": StudyDesign.SYSTEMATIC_REVIEW,
            "randomized controlled trial": StudyDesign.RCT,
            "controlled clinical trial": StudyDesign.RCT,
            "clinical trial": StudyDesign.RCT,
            "rct": StudyDesign.RCT,
            "cohort": StudyDesign.COHORT,
            "cohort studies": StudyDesign.COHORT,
            "observational study": StudyDesign.COHORT,
            "case-control": StudyDesign.CASE_CONTROL,
            "case control": StudyDesign.CASE_CONTROL,
            "case-control studies": StudyDesign.CASE_CONTROL,
            "case series": StudyDesign.CASE_SERIES,
            "case reports": StudyDesign.CASE_SERIES,
            "case report": StudyDesign.CASE_SERIES,
        }
        normalized = str(raw).lower().replace("-", " ").strip()
        return mapping.get(normalized, StudyDesign.OTHER)

    def _as_dict(self, article_dict: Optional[Dict[str, Any]], abstract: Optional[str]) -> Dict[str, Any]:
        if article_dict is None:
            return {"abstract": abstract or ""}
        if hasattr(article_dict, "model_dump"):
            return article_dict.model_dump()
        if hasattr(article_dict, "dict"):
            return article_dict.dict()
        return dict(article_dict)

    def _as_list(self, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(v) for v in value if v is not None]
        if isinstance(value, tuple):
            return [str(v) for v in value if v is not None]
        return [str(value)]

    def _extract_year(self, article_dict: Dict[str, Any]) -> Optional[int]:
        raw_year = article_dict.get("year") or article_dict.get("publication_year")
        if raw_year is None:
            return None
        try:
            return int(raw_year)
        except (TypeError, ValueError):
            match = re.search(r"\d{4}", str(raw_year))
            return int(match.group(0)) if match else None

    def _design_from_xml(self, article_dict: Dict[str, Any]) -> Optional[StudyDesign]:
        publication_types = self._as_list(article_dict.get("publication_types"))
        study_types = self._as_list(article_dict.get("study_types"))
        mesh_terms = self._as_list(article_dict.get("mesh_terms"))

        # Highest-certainty official PubMed publication types first.
        priority_rules = [
            (StudyDesign.META_ANALYSIS, ("meta-analysis", "meta analysis")),
            (StudyDesign.SYSTEMATIC_REVIEW, ("systematic review",)),
            (StudyDesign.RCT, ("randomized controlled trial", "controlled clinical trial")),
            (StudyDesign.COHORT, ("cohort study", "cohort studies", "observational study")),
            (StudyDesign.CASE_CONTROL, ("case-control", "case control")),
            (StudyDesign.CASE_SERIES, ("case reports", "case report", "case series")),
        ]

        pub_text = " | ".join(publication_types + study_types).lower()
        for design, needles in priority_rules:
            if any(needle in pub_text for needle in needles):
                return design

        # MeSH is also XML-grounded, but less direct than PublicationType.
        mesh_text = " | ".join(mesh_terms).lower()
        for design, needles in priority_rules[3:]:
            if any(needle in mesh_text for needle in needles):
                return design
        return None

    def _extract_p_value_present(self, abstract: str) -> bool:
        return bool(re.search(r"\bp(?:-value)?\s*[<=>]\s*(?:0?\.\d+|\d+(?:\.\d+)?)\b", abstract, re.IGNORECASE))

    def _extract_ci_present(self, abstract: str) -> bool:
        return bool(
            re.search(r"\b95\s*%\s*(?:ci|confidence interval)\b", abstract, re.IGNORECASE)
            or re.search(r"\bconfidence intervals?\b", abstract, re.IGNORECASE)
        )

    def _extract_preregistration_id(self, abstract: str) -> Optional[str]:
        match = re.search(r"\b(NCT\d{8}|ISRCTN\d{8})\b", abstract, re.IGNORECASE)
        return match.group(1).upper() if match else None

    def _refine_sample_size(self, abstract: str, current_n: Optional[int], design: StudyDesign) -> Optional[int]:
        if current_n and current_n > 0:
            return current_n

        # Robust regex for sample size
        patterns = [
            r"\bn\s*=\s*([\d,]{1,9})\b",
            r"\btotal of\s*([\d,]{1,9})\s*(?:patients|participants|subjects)\b",
            r"\b([\d,]{1,9})\s*(?:patients|participants|subjects)\b",
            r"\bpooled sample of\s*([\d,]{1,9})\b",
            r"\benrolled\s*([\d,]{1,9})\b",
        ]
        for pat in patterns:
            match = re.search(pat, abstract, re.IGNORECASE)
            if match:
                return int(match.group(1).replace(",", ""))

        # Conservative default for meta-analyses
        if design == StudyDesign.META_ANALYSIS:
            return 100
        return current_n

    async def _infer_design_with_llm(self, abstract: str, pmid: Optional[str]) -> Optional[StudyDesign]:
        if self.llm is None or not abstract.strip():
            return None

        try:
            chain = self.prompt | self.llm | self.parser
            parsed = await safe_ainvoke(chain, {"abstract": abstract[:2000]})
            if isinstance(parsed, StudyMetadata):
                return self._normalize_design(str(parsed.study_design))
            if isinstance(parsed, dict):
                return self._normalize_design(parsed.get("study_design"))
        except Exception as e:
            logger.warning("LLM metadata fallback failed for PMID %s: %s", pmid, e)
        return None

    async def extract(
        self,
        abstract: Optional[str] = None,
        pmid: Optional[str] = None,
        article_dict: Optional[Dict[str, Any]] = None,
    ) -> StudyMetadata:
        """Extract metadata via XML truth, regex math, and LLM design fallback."""
        try:
            article = self._as_dict(article_dict, abstract)
            abstract_text = str(article.get("abstract") or abstract or "")
            publication_types = self._as_list(article.get("publication_types") or article.get("study_types"))
            mesh_terms = self._as_list(article.get("mesh_terms"))

            design = self._design_from_xml(article)
            if design is None:
                design = await self._infer_design_with_llm(abstract_text, pmid)
            if design is None:
                design = StudyDesign.OTHER

            metadata = StudyMetadata(
                sample_size=self._refine_sample_size(abstract_text, None, design),
                study_design=design,
                has_p_value=self._extract_p_value_present(abstract_text),
                has_CI=self._extract_ci_present(abstract_text),
                preregistration_id=self._extract_preregistration_id(abstract_text),
                year=self._extract_year(article),
                source_type=str(article.get("source") or "pubmed").lower(),
                mesh_terms=mesh_terms,
                publication_types=publication_types,
            )

            if self._debug_enabled() and pmid:
                self.debug_manager.save_json(
                    f"pubmed/{pmid}_metadata_llm.json",
                    self._llm_metadata_payload(metadata),
                )

            return metadata
        except Exception as e:
            if self._debug_enabled():
                if pmid:
                    self.debug_manager.save_exception(f"exceptions/metadata_{pmid}_exception.txt", e)
                else:
                    self.debug_manager.save_exception("exceptions/metadata_unknown_exception.txt", e)
            raise MetadataExtractionError(f"Extraction failed for PMID {pmid}: {e}")

    async def extract_batch(self, abstracts: List[Dict[str, str]]) -> List[StudyMetadata]:
        """Extract metadata for a batch of abstracts."""
        import asyncio

        tasks = [self.extract(article_dict=item, pmid=item.get("pmid")) for item in abstracts]
        return await asyncio.gather(*tasks)
