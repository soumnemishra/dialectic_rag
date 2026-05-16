from typing import Dict, List
import math


class HybridRanker:
    """Hybrid ranker that performs RRF fusion and final scoring.

    Expects dicts mapping pmid->score for BM25 and dense, and a rerank score dict.
    """

    def __init__(self, k_rrf: int = 60, lambda_temporal: float = 0.15, study_priors: Dict[str, float] = None):
        self.k_rrf = k_rrf
        self.lambda_temporal = lambda_temporal
        self.study_priors = study_priors or {}

    def rrf(self, rankings: List[List[str]]) -> Dict[str, float]:
        # rankings: list of lists of pmids in ranked order (best first)
        scores = {}
        for ranking in rankings:
            for i, pmid in enumerate(ranking, start=1):
                scores.setdefault(pmid, 0.0)
                scores[pmid] += 1.0 / (self.k_rrf + i)
        return scores

    def normalize(self, d: Dict[str, float]) -> Dict[str, float]:
        if not d:
            return {}
        mx = max(d.values())
        if mx == 0:
            return {k: 0.0 for k in d}
        return {k: v / mx for k, v in d.items()}

    def compute_final_scores(self, bm25_scores: Dict[str, float], dense_scores: Dict[str, float], rerank_scores: Dict[str, float], years: Dict[str, int], study_type: Dict[str, str], current_year: int = 2026, contradiction_bonus: Dict[str, float] = None) -> Dict[str, float]:
        # Normalize components
        bm_n = self.normalize(bm25_scores)
        dn_n = self.normalize(dense_scores)
        rr_n = self.normalize(rerank_scores)

        contradiction_bonus = contradiction_bonus or {}

        final = {}
        for pmid in set(list(bm_n.keys()) + list(dn_n.keys()) + list(rr_n.keys())):
            bm = bm_n.get(pmid, 0.0)
            dn = dn_n.get(pmid, 0.0)
            rr = rr_n.get(pmid, 0.0)
            year = years.get(pmid)
            age = (current_year - year) if year else 10
            temporal = math.exp(-self.lambda_temporal * age)
            stype = study_type.get(pmid, "").lower()
            prior = self.study_priors.get(stype, 1.0)
            cbonus = contradiction_bonus.get(pmid, 0.0)

            score = 0.20 * bm + 0.20 * dn + 0.30 * rr + 0.10 * temporal + 0.10 * prior + 0.10 * cbonus
            final[pmid] = score
        return final
