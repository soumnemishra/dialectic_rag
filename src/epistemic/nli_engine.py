import logging
import asyncio
from typing import Dict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from src.core.registry import ModelRegistry, safe_ainvoke
from src.config import settings

logger = logging.getLogger(__name__)

class NLIEngine:
    """Neural model for Natural Language Inference (ENTAILMENT | CONTRADICTION | NEUTRAL)."""
    
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or getattr(settings, "NLI_MODEL_NAME", "cross-encoder/nli-deberta-v3-base")
        try:
            self.cross_model = ModelRegistry.get_cross_encoder(self.model_name)
        except Exception:
            self.cross_model = None

        self.llm = ModelRegistry.get_flash_llm(temperature=0.0, json_mode=True)
        
    async def classify(self, premise: str, hypothesis: str) -> Dict[str, Any]:
        """Classify relation between premise and hypothesis."""
        # Try local cross-encoder first
        if self.cross_model:
            try:
                # Mocking logic for local model if it's integrated via registry
                # Assuming registry returns a model with .predict()
                raw = await asyncio.to_thread(self.cross_model.predict, [[premise, hypothesis]])
                # Map logits to labels
                if len(raw[0]) == 3: # Assuming 3-class logits [entailment, neutral, contradiction]
                    import numpy as np
                    probs = np.exp(raw[0]) / np.sum(np.exp(raw[0]))
                    idx = int(np.argmax(probs))
                    labels = ["ENTAILMENT", "NEUTRAL", "CONTRADICTION"]
                    probs_list = [float(x) for x in probs.tolist()]
                    probs_map = {lab: round(probs_list[i], 4) for i, lab in enumerate(labels)}
                    return {
                        "label": labels[idx],
                        "confidence": round(float(probs_list[idx]), 3),
                        "probs": probs_map,
                        "probs_array": probs_list,
                    }
            except Exception as e:
                logger.warning(f"NLI local model failed: {e}")

        # LLM fallback
        if self.llm:
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an NLI classifier. Return JSON: {{\"label\": \"ENTAILMENT|CONTRADICTION|NEUTRAL\", \"confidence\": 0.0}}"),
                ("human", f"Premise: {premise}\nHypothesis: {hypothesis}")
            ])
            try:
                res = await safe_ainvoke(prompt | self.llm | JsonOutputParser(), {})
                return res
            except Exception as e:
                logger.error(f"NLI LLM fallback failed: {e}")

        return {"label": "NEUTRAL", "confidence": 0.0}
