import logging
import re
from typing import List, Dict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from src.core.registry import ModelRegistry, safe_ainvoke
from src.models.schemas import StudyMetadata
from src.prompts.templates import with_json_system_suffix

logger = logging.getLogger(__name__)

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
        self.llm = ModelRegistry.get_flash_llm(temperature=0.0, json_mode=True)
        if self.llm is None:
            raise RuntimeError("MetadataExtractor: Flash LLM unavailable")
        self.parser = JsonOutputParser(pydantic_object=StudyMetadata)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", with_json_system_suffix(METADATA_EXTRACTION_SYSTEM_PROMPT)),
            ("human", METADATA_EXTRACTION_HUMAN_PROMPT),
        ])

    def _normalize_design(self, raw: Optional[str]) -> str:
        if not raw:
            return "other"
        mapping = {
            "meta-analysis": "meta_analysis",
            "meta analysis": "meta_analysis",
            "systematic review": "systematic_review",
            "randomized controlled trial": "rct",
            "rct": "rct",
            "cohort": "cohort",
            "case-control": "case_control",
            "case control": "case_control",
            "case series": "case_series",
        }
        normalized = str(raw).lower().replace("-", " ").strip()
        return mapping.get(normalized, "other")

    def _refine_sample_size(self, abstract: str, current_n: Optional[int], design: str) -> Optional[int]:
        if current_n and current_n > 0:
            return current_n
        
        # Robust regex for sample size
        patterns = [
            r"n\s*=\s*(\d{1,7})",
            r"total of\s*(\d{1,7})\s*patients",
            r"(\d{1,7})\s*participants",
            r"pooled sample of\s*(\d{1,7})",
            r"enrolled\s*(\d{1,7})"
        ]
        for pat in patterns:
            match = re.search(pat, abstract, re.IGNORECASE)
            if match:
                return int(match.group(1))
        
        # Conservative default for meta-analyses
        if design == "meta_analysis":
            return 100
        return current_n

    async def extract(self, abstract: str, pmid: Optional[str] = None) -> StudyMetadata:
        """Extract metadata from a single abstract."""
        try:
            # Smart trim if needed
            abstract_text = str(abstract)[:2000]
            # Include parser in the chain for automatic parsing of AIMessage
            chain = self.prompt | self.llm | self.parser
            
            parsed_dict = await safe_ainvoke(chain, {"abstract": abstract_text})
            
            # If parsed_dict is already a StudyMetadata object (because of parser)
            if isinstance(parsed_dict, StudyMetadata):
                parsed = parsed_dict
                # Normalize design and enforce boolean
                parsed.study_design = self._normalize_design(str(parsed.study_design))
                parsed.has_p_value = bool(parsed.has_p_value)
                parsed.has_CI = bool(getattr(parsed_dict, "has_CI", getattr(parsed_dict, "ci_present", False)))
            else:
                # Manually map fields if it's a dict
                year = parsed_dict.get("publication_year") or parsed_dict.get("year")
                raw_design = parsed_dict.get("study_design")
                
                parsed = StudyMetadata(
                    sample_size=parsed_dict.get("sample_size"),
                    study_design=self._normalize_design(raw_design),
                    has_p_value=bool(parsed_dict.get("p_value_present", False)),
                    has_CI=bool(parsed_dict.get("ci_present", False)),
                    preregistration_id=parsed_dict.get("preregistration_id"),
                    year=year,
                    source_type="pubmed"
                )
            
            # Refine sample size with regex
            parsed.sample_size = self._refine_sample_size(abstract, parsed.sample_size, str(parsed.study_design))
            
            # PMID belongs to EvidenceItem, not StudyMetadata
            return parsed
        except Exception as e:
            logger.error(f"Metadata extraction failed for PMID {pmid}: {e}")
            # Return empty metadata on failure to avoid crashing
            return StudyMetadata(year=None, source_type="pubmed")

    async def extract_batch(self, abstracts: List[Dict[str, str]]) -> List[StudyMetadata]:
        """Extract metadata for a batch of abstracts."""
        import asyncio
        tasks = [self.extract(item["abstract"], item.get("pmid")) for item in abstracts]
        return await asyncio.gather(*tasks)
