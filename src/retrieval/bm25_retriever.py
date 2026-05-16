from typing import List, Dict, Any
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class BM25Retriever:
    """Simple BM25 retriever over in-memory documents.

    Each document is expected to be a dict with keys: 'pmid', 'title', 'abstract', 'mesh_terms'
    """

    def __init__(self):
        try:
            from rank_bm25 import BM25Okapi
            self._bm25_impl = BM25Okapi
            self.available = True
        except Exception:
            # Fallback: we'll use sklearn Tfidf + cosine similarity as proxy
            self._bm25_impl = None
            self.available = False

    def _tokenize(self, text: str) -> List[str]:
        return [t.lower() for t in (text or "").split() if t.strip()]

    def fit(self, docs: List[Dict[str, Any]]):
        # Build corpus over title+abstract+mesh
        self.docs = docs
        self.corpus = []
        for d in docs:
            mesh = " ".join(d.get("mesh_terms", []) or [])
            txt = " ".join([d.get("title", ""), d.get("abstract", ""), mesh])
            self.corpus.append(self._tokenize(txt))

        if self.available:
            self.bm25 = self._bm25_impl(self.corpus)
        else:
            # sklearn fallback
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vectorizer = TfidfVectorizer().fit([" ".join(c) for c in self.corpus])
            self.tfidf_matrix = self.vectorizer.transform([" ".join(c) for c in self.corpus])

    def score(self, query: str) -> Dict[str, float]:
        q_tok = self._tokenize(query)
        if self.available:
            scores = self.bm25.get_scores(q_tok)
            return {str(d.get("pmid")): float(score) for d, score in zip(self.docs, scores)}
        else:
            qtxt = " ".join(q_tok)
            qvec = self.vectorizer.transform([qtxt])
            from sklearn.metrics.pairwise import cosine_similarity
            sims = cosine_similarity(qvec, self.tfidf_matrix).flatten()
            return {str(d.get("pmid")): float(s) for d, s in zip(self.docs, sims)}
