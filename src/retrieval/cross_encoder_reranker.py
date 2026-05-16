import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Cross-encoder re-ranker using sentence-transformers CrossEncoder when available.

    Fallback: uses a simple concatenation + dense similarity if CrossEncoder unavailable.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        try:
            from sentence_transformers import CrossEncoder
            self.CrossEncoder = CrossEncoder
            self.model = CrossEncoder(model_name)
            self.available = True
        except Exception:
            self.model = None
            self.available = False

    def rerank(self, query: str, docs: List[Dict], top_k: int = 50) -> Dict[str, float]:
        """Return dict pmid->score for provided docs (docs are dicts with title+abstract)."""
        texts = []
        pmids = []
        for d in docs:
            pmid = str(d.get("pmid"))
            txt = (d.get("title", "") or "") + " \n " + (d.get("abstract", "") or "")
            pmids.append(pmid)
            texts.append(txt)

        scores = {}
        try:
            if self.available and self.model is not None:
                pairs = [[query, t] for t in texts]
                out = self.model.predict(pairs)
                for pmid, s in zip(pmids, out):
                    scores[pmid] = float(s)
            else:
                # Fallback: encode with SentenceTransformer and compute cosine similarity
                from sentence_transformers import SentenceTransformer
                from sklearn.metrics.pairwise import cosine_similarity
                emb = SentenceTransformer("all-MiniLM-L6-v2")
                qv = emb.encode([query], convert_to_numpy=True)
                dv = emb.encode(texts, convert_to_numpy=True)
                sims = cosine_similarity(qv, dv).flatten()
                for pmid, s in zip(pmids, sims):
                    scores[pmid] = float(s)
        except Exception as e:
            logger.debug("CrossEncoder rerank failed: %s", e)
            # fallback to zeros
            for pmid in pmids:
                scores[pmid] = 0.0

        return scores
