from __future__ import annotations
import logging
from typing import Optional, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from src.core.registry import ModelRegistry, safe_ainvoke
from src.models.schemas import PICO

logger = logging.getLogger(__name__)

PICO_EXTRACTION_PROMPT = """
You are a clinical research assistant. 
Decompose the clinical question into PICO components, classify the clinical intent, and assess the risk level of the clinical scenario.

Clinical Intent Categories:
- Diagnostic: Focuses on determining the cause or presence of a condition.
- Therapeutic: Focuses on the effects of a treatment or intervention.
- Prognostic: Focuses on the likely course or outcome of a condition.
- Preventive: Focuses on preventing the onset or recurrence of a condition.

IMPORTANT: Ignore any multiple-choice options provided at the end of the question. Extract components based ONLY on the patient's clinical vignette.

Return ONLY a JSON object.

Question: {question}

JSON keys:
- intent: str ("Diagnostic", "Therapeutic", "Prognostic", or "Preventive")
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

    async def extract(self, question: str) -> dict[str, Any]:
        """Extract PICO components from a clinical question."""
        try:
            pico_res = await safe_ainvoke(self.prompt | self.llm | JsonOutputParser(), {"question": question})
            if isinstance(pico_res, PICO):
                return pico_res.model_dump()
            if isinstance(pico_res, dict):
                return pico_res
            if hasattr(pico_res, "dict"):
                return pico_res.dict()
            return PICO(intent="Therapeutic", population="unknown", intervention="unknown", outcome="unknown", risk_level="unknown").model_dump()
        except Exception as e:
            logger.error(f"PICO extraction failed: {e}")
            # Fallback to empty PICO to avoid breaking the pipeline
            return PICO(intent="Therapeutic", population="unknown", intervention="unknown", outcome="unknown", risk_level="unknown").model_dump()

    async def generate_diagnostic_hypotheses(self, vignette: str) -> list[str]:
        """Generate 3-5 diagnostic hypotheses for a clinical vignette."""
        prompt = ChatPromptTemplate.from_template("""
You are a senior clinical diagnostician. Based on the following clinical vignette, generate 3-5 distinct diagnostic hypotheses that could explain the patient's symptoms.
Each hypothesis should be a concise medical condition or cause.

Vignette: {vignette}

Return ONLY a JSON list of strings.
""")
        try:
            res = await safe_ainvoke(prompt | self.llm | JsonOutputParser(), {"vignette": vignette})
            if isinstance(res, list):
                return res
            return ["Diagnosis 1", "Diagnosis 2"] # Fallback
        except Exception as e:
            logger.error(f"Diagnostic hypothesis generation failed: {e}")
            return ["Diagnosis 1", "Diagnosis 2"]
