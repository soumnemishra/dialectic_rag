import logging
import yaml
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
import re
from src.models.schemas import PICO
from src.core.registry import ModelRegistry

logger = logging.getLogger(__name__)

class ApplicabilityScorer:
    """
    Compares patient PICO with study PICO using embedding similarity.
    Includes outcome-type discounting for clinical validity.
    """

    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.weights = self.config.get("applicability", {})
        self.model_name = self.weights.get("embedding_model", "all-MiniLM-L6-v2")
        # Use get_sentence_transformer instead of get_embedding_model
        self.encoder = ModelRegistry.get_sentence_transformer(self.model_name)

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        path = config_path or Path(__file__).resolve().parents[1] / "config" / "default.yaml"
        try:
            with open(path, "r") as f:
                return yaml.safe_load(f)
        except Exception:
            return {
                "applicability": {
                    "w_population": 0.25,
                    "w_intervention": 0.35,
                    "w_comparator": 0.15,
                    "w_outcome": 0.25,
                    "surrogate_outcome_discount": 0.7
                }
            }

    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        if v1 is None or v2 is None:
            return 0.0
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(v1, v2) / (norm1 * norm2))

    def _token_overlap(self, left_text: str, right_text: str) -> float:
        left_tokens = set(re.findall(r"[a-z]{3,}", (left_text or "").lower()))
        right_tokens = set(re.findall(r"[a-z]{3,}", (right_text or "").lower()))
        if not left_tokens or not right_tokens:
            return 0.0
        shared = left_tokens & right_tokens
        return len(shared) / max(1, min(len(left_tokens), len(right_tokens)))

    def compute(
        self,
        patient_pico: PICO,
        study_abstract: Optional[str] = None,
        study_population_profile: Optional[str] = None,
    ) -> float:
        """
        Compute weighted applicability score using embedding similarity.
        Compares the query's PICO against the study population profile when available,
        otherwise falls back to the study abstract.
        applicability = 0.7 * cos_sim(pop_emb, study_emb) + 0.3 * cos_sim(int_out_emb, study_emb).
        """
        if not self.encoder:
            logger.warning("Embedding model unavailable for ApplicabilityScorer.")
            return 0.5
            
        if not study_abstract:
            return 0.5
            
        try:
            # Prepare patient representations
            pop_text = f"Population: {patient_pico.population}"
            int_out_text = f"Intervention/Setting: {patient_pico.intervention} Outcome: {patient_pico.outcome}"
            
            # Use a population-specific profile if available; otherwise fall back to the abstract.
            study_text = (study_population_profile or study_abstract or "")[:2000]
            
            pop_emb = self.encoder.encode(pop_text)
            int_out_emb = self.encoder.encode(int_out_text)
            study_emb = self.encoder.encode(study_text)
            
            pop_sim = self._cosine_similarity(pop_emb, study_emb)
            int_out_sim = self._cosine_similarity(int_out_emb, study_emb)
            lexical_overlap = self._token_overlap(
                f"{patient_pico.population} {patient_pico.intervention} {patient_pico.outcome}",
                study_text,
            )
            
            raw_applicability = (0.5 * pop_sim) + (0.2 * int_out_sim) + (0.3 * lexical_overlap)
            
            # Calibrate the soft similarity score into a usable applicability weight.
            applicability = 0.3 + (0.7 * max(0.0, min(1.0, float(raw_applicability))))
            
            # Clip to [0, 1]
            return max(0.0, min(1.0, float(applicability)))
        except Exception as e:
            logger.warning(f"Embedding failed for ApplicabilityScorer: {e}")
            return 0.5
