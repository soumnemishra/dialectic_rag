import asyncio
import logging
from typing import Dict, Any

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from src.core.registry import ModelRegistry, safe_ainvoke
from src.config import settings

logger = logging.getLogger(__name__)


class NLIResult(BaseModel):
    label: str = Field(description="One of: ENTAILMENT, CONTRADICTION, NEUTRAL")
    confidence: float = Field(description="Confidence between 0.0 and 1.0")


_NLI_SYSTEM_PROMPT = (
    "You are a concise Natural Language Inference classifier. "
    "Given a PREMISE and a HYPOTHESIS, decide whether the PREMISE entails, contradicts, or is neutral with respect to the HYPOTHESIS. "
    "Return ONLY a JSON object with keys: label (ENTAILMENT|CONTRADICTION|NEUTRAL) and confidence (0.0-1.0)."
)

_NLI_HUMAN_PROMPT = """
Premise: {premise}

Hypothesis: {hypothesis}

Respond with JSON: {{"label": "ENTAILMENT|CONTRADICTION|NEUTRAL", "confidence": 0.0}}
"""


class NliAgent:
    """NLI agent with optional Cross-Encoder fast path and LLM fallback."""

    def __init__(self, model_name: str | None = None):
        from src.config import settings
        self.model_name = model_name or getattr(settings, "NLI_MODEL_NAME", "cross-encoder/nli-deberta-v3-base")
        # Try to load a cross-encoder NLI model (fast, local) — may be None
        try:
            self.cross_model = ModelRegistry.get_cross_encoder(self.model_name)
        except Exception:
            self.cross_model = None

        # LLM fallback (fast/light LLM)
        try:
            self.llm = ModelRegistry.get_flash_llm(temperature=0.0, json_mode=True)
        except Exception:
            self.llm = None

        if self.llm is not None:
            self.parser = JsonOutputParser(pydantic_object=NLIResult)
            self.prompt = ChatPromptTemplate.from_messages([
                ("system", _NLI_SYSTEM_PROMPT),
                ("human", _NLI_HUMAN_PROMPT),
            ])

        logger.info("NliAgent initialized | cross_encoder=%s llm=%s | model_name=%s", bool(self.cross_model), bool(self.llm), self.model_name)

    async def classify(self, premise: str, hypothesis: str) -> Dict[str, Any]:
        """Return {'label': str, 'confidence': float}.

        Tries cross-encoder fast path first; falls back to LLM-based JSON parsing.
        """
        premise = str(premise or "").strip()
        hypothesis = str(hypothesis or "").strip()

        # Fast Cross-Encoder path (blocking model, run in thread)
        if self.cross_model is not None:
            try:
                raw = await asyncio.to_thread(self.cross_model.predict, [[premise, hypothesis]])
                # raw can be scalar, list, or nested list depending on model
                if isinstance(raw, (list, tuple)):
                    vals = raw[0] if len(raw) == 1 and isinstance(raw[0], (list, tuple)) else raw
                    # If we have 3-class logits, map argmax -> label
                    if isinstance(vals, (list, tuple)) and len(vals) == 3:
                        idx = int(max(range(len(vals)), key=lambda i: float(vals[i] or 0)))
                        mapping = {0: "ENTAILMENT", 1: "NEUTRAL", 2: "CONTRADICTION"}
                        label = mapping.get(idx, "NEUTRAL")
                        # Confidence via softmax approximation
                        try:
                            import math

                            exps = [math.exp(float(v)) for v in vals]
                            s = sum(exps) if sum(exps) != 0 else 1.0
                            conf = float(exps[idx] / s)
                        except Exception:
                            conf = 0.6
                        return {"label": label, "confidence": round(min(max(conf, 0.0), 1.0), 3)}

                    # Scalar score fallback → interpret sign
                    try:
                        score = float(vals[0]) if isinstance(vals, (list, tuple)) else float(vals)
                        if score > 0.3:
                            return {"label": "ENTAILMENT", "confidence": round(min(score, 1.0), 3)}
                        if score < -0.3:
                            return {"label": "CONTRADICTION", "confidence": round(min(abs(score), 1.0), 3)}
                        return {"label": "NEUTRAL", "confidence": 0.5}
                    except Exception:
                        pass
            except Exception as exc:
                logger.debug("NLI cross-encoder failed: %s", exc)

        # LLM fallback (skipped in FAST_EPISTEMIC to reduce latency)
        if self.llm is not None and not settings.FAST_EPISTEMIC:
            try:
                chain = self.prompt | self.llm | self.parser
                resp = await safe_ainvoke(chain, {"premise": premise, "hypothesis": hypothesis})
                label = str(resp.get("label", "NEUTRAL")).strip().upper()
                conf = float(resp.get("confidence", 0.0) or 0.0)
                if label not in {"ENTAILMENT", "CONTRADICTION", "NEUTRAL"}:
                    label = "NEUTRAL"
                return {"label": label, "confidence": round(min(max(conf, 0.0), 1.0), 3)}
            except Exception as exc:
                logger.warning("NLI LLM fallback failed: %s", exc)
        else:
            if settings.FAST_EPISTEMIC:
                logger.debug("FAST_EPISTEMIC enabled: skipping NLI LLM fallback")

        # Final conservative default
        return {"label": "NEUTRAL", "confidence": 0.0}
