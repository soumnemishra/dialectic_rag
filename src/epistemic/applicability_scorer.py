"""Applicability Scorer — Config-Driven Embedding Similarity.

Computes how applicable a retrieved study is to the patient's clinical
context by comparing PICO-derived embeddings against the study text.

Mathematical formulation (aligned with thesis §Applicability):
    A_raw = w_pop · cos(p, s) + w_int · cos(io, s) + w_lex · overlap(Q, S)
    A     = 0.3 + 0.7 · clamp(A_raw, 0, 1)

All weights are loaded from ``config/default.yaml → applicability``.
"""

import logging
import re
from typing import Optional, Dict, Any

import numpy as np

from src.models.schemas import PICO
from src.core.registry import ModelRegistry
from src.config import epistemic_settings

logger = logging.getLogger(__name__)


class ApplicabilityScoringError(Exception):
    pass


class ApplicabilityScorer:
    """Compares patient PICO with study PICO using embedding similarity.

    All tunable weights are loaded from the ``applicability`` section of
    ``config/default.yaml``.  The scorer validates at init time that the
    three scoring weights sum to 1.0 (±tolerance).

    Args:
        config: Optional override dict; defaults to the centralised
            ``epistemic_settings`` singleton.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config if config is not None else epistemic_settings
        app_cfg = self.config.get("applicability", {})

        # --- Config-driven weights (F1 fix) ---
        self.w_population: float = float(app_cfg.get("w_population", 0.50))
        self.w_intervention: float = float(app_cfg.get("w_intervention", 0.20))
        self.w_lexical: float = float(app_cfg.get("w_lexical", 0.30))

        # --- Weight validation ---
        weight_sum = self.w_population + self.w_intervention + self.w_lexical
        if abs(weight_sum - 1.0) > 0.01:
            raise ApplicabilityScoringError(
                f"Applicability weights must sum to 1.0, got {weight_sum:.4f} "
                f"(w_population={self.w_population}, w_intervention={self.w_intervention}, "
                f"w_lexical={self.w_lexical})"
            )

        logger.info(
            "ApplicabilityScorer initialised: w_pop=%.2f, w_int=%.2f, w_lex=%.2f",
            self.w_population, self.w_intervention, self.w_lexical,
        )

        self.model_name: str = app_cfg.get("embedding_model", "all-MiniLM-L6-v2")
        self.encoder = ModelRegistry.get_sentence_transformer(self.model_name)

    # ------------------------------------------------------------------
    # Maths helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        """Cosine similarity with zero-vector safety."""
        if v1 is None or v2 is None:
            return 0.0
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(v1, v2) / (norm1 * norm2))

    @staticmethod
    def _token_overlap(left_text: str, right_text: str) -> float:
        """Jaccard-style lexical overlap on 3+ character tokens."""
        left_tokens = set(re.findall(r"[a-z]{3,}", (left_text or "").lower()))
        right_tokens = set(re.findall(r"[a-z]{3,}", (right_text or "").lower()))
        if not left_tokens or not right_tokens:
            return 0.0
        shared = left_tokens & right_tokens
        return len(shared) / max(1, min(len(left_tokens), len(right_tokens)))

    # ------------------------------------------------------------------
    # Main scoring
    # ------------------------------------------------------------------

    def compute(
        self,
        patient_pico: PICO,
        study_abstract: Optional[str] = None,
        study_population_profile: Optional[str] = None,
    ) -> float:
        """Compute weighted applicability score using embedding similarity.

        Formula (config-driven):
            A_raw = w_pop · cos(pop_emb, study_emb)
                  + w_int · cos(int_out_emb, study_emb)
                  + w_lex · token_overlap(PICO_text, study_text)
            A     = 0.3 + 0.7 · clamp(A_raw, 0, 1)

        Args:
            patient_pico: The query-side PICO object.
            study_abstract: Full abstract of the study.
            study_population_profile: Optional population-specific profile
                (from claim extraction) for higher-precision matching.

        Returns:
            Applicability score in [0, 1].

        Raises:
            ApplicabilityScoringError: If embedding model is unavailable or
                embedding computation fails.
        """
        if not self.encoder:
            raise ApplicabilityScoringError(
                "Embedding model unavailable for ApplicabilityScorer."
            )

        if not study_abstract:
            return 0.0

        try:
            # Prepare patient representations
            population = patient_pico.population or "Unknown"
            intervention = patient_pico.intervention or "Unknown"
            outcome = patient_pico.outcome or "Unknown"

            pop_text = f"Population: {population}"
            int_out_text = f"Intervention/Setting: {intervention} Outcome: {outcome}"

            # Use a population-specific profile if available; otherwise the abstract.
            study_text = (study_population_profile or study_abstract or "")[:2000]

            # Encode
            pop_emb = self.encoder.encode(pop_text)
            int_out_emb = self.encoder.encode(int_out_text)
            study_emb = self.encoder.encode(study_text)

            # Sub-scores
            pop_sim = self._cosine_similarity(pop_emb, study_emb)
            int_out_sim = self._cosine_similarity(int_out_emb, study_emb)
            lexical_overlap = self._token_overlap(
                f"{population} {intervention} {outcome}",
                study_text,
            )

            # Weighted combination — all weights from YAML
            raw_applicability = (
                self.w_population * pop_sim
                + self.w_intervention * int_out_sim
                + self.w_lexical * lexical_overlap
            )

            # Calibrate the soft similarity score into a usable weight.
            # Floor of 0.3 ensures no study is completely zeroed out.
            applicability = 0.3 + (0.7 * max(0.0, min(1.0, float(raw_applicability))))

            return max(0.0, min(1.0, float(applicability)))

        except Exception as e:
            raise ApplicabilityScoringError(
                f"Embedding failed for ApplicabilityScorer: {e}"
            )
