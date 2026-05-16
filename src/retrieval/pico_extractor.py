import logging
from typing import Optional, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from src.core.registry import ModelRegistry, safe_ainvoke
from src.models.schemas import PICO

logger = logging.getLogger(__name__)

PICO_EXTRACTION_PROMPT = """
You are a clinical research assistant. 
Decompose the clinical question into PICO components, and assess the risk level of the clinical scenario.
IMPORTANT: Ignore any multiple-choice options provided at the end of the question. Extract PICO components based ONLY on the patient's clinical vignette.

Return ONLY a JSON object.

Question: {question}

JSON keys:
- population: str (e.g., "adults with type 2 diabetes")
- intervention: str (e.g., "metformin" or the diagnostic test)
- comparator: str or null (e.g., "placebo" or "standard care")
- outcome: str (e.g., "all-cause mortality")
- risk_level: str ("low", "medium", "high" based on acuity and severity)
"""

class PICOExtractor:
    def __init__(self):
        self.llm = ModelRegistry.get_flash_llm(temperature=0.0, json_mode=True)
        self.prompt = ChatPromptTemplate.from_template(PICO_EXTRACTION_PROMPT)

    async def extract(self, question: str) -> Dict[str, Any]:
        """Extract PICO components from a clinical question."""
        try:
            pico_res = await safe_ainvoke(self.prompt | self.llm, {"question": question})
            if isinstance(pico_res, PICO):
                return pico_res.model_dump()
            if isinstance(pico_res, dict):
                return pico_res
            if hasattr(pico_res, "dict"):
                return pico_res.dict()
            return PICO(population="unknown", intervention="unknown", outcome="unknown", risk_level="unknown").model_dump()
        except Exception as e:
            logger.error(f"PICO extraction failed: {e}")
            # Fallback to empty PICO to avoid breaking the pipeline
            return PICO(population="unknown", intervention="unknown", outcome="unknown", risk_level="unknown").model_dump()
