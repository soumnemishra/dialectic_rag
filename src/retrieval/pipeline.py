from typing import List, Dict, Any
from src.retrieval.query_generator import QueryGenerator, PICO
from src.pubmed_client import PubMedClient
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_ranker import HybridRanker
from src.retrieval.mmr import mmr_select
from src.retrieval.cross_encoder_reranker import CrossEncoderReranker
from src.retrieval.mesh_expander import MeshExpander
from src.utils.debug_utils import get_debug_manager
import logging

logger = logging.getLogger(__name__)


class RetrievalPipeline:
    def __init__(self, config: Dict[str, Any] = None):
        self.pubmed = PubMedClient()
        # provide a MeshExpander instance to the QueryGenerator
        mesh = MeshExpander()
        self.qgen = QueryGenerator(mesh_expander=mesh)
        self.bm25 = BM25Retriever()
        self.hybrid = HybridRanker()
        self.reranker = CrossEncoderReranker()
        self.debug = get_debug_manager()
        self.config = config or {}

    async def run_for_hypothesis(self, claim: str, pico: Dict[str, Any], hypothesis: str, top_k: int = 20) -> Dict[str, Any]:
        pico_obj = PICO(
            population=pico.get("population") if pico else None,
            intervention=pico.get("intervention") if pico else None,
            comparator=pico.get("comparator") if pico else None,
            outcome=pico.get("outcome") if pico else None,
            intent=pico.get("intent") if pico else None,
            risk_level=pico.get("risk_level") if pico else None,
        )

        queries = self.qgen.generate(claim=claim, pico=pico_obj, hypothesis=hypothesis)

        # Retrieve up to 200 PMIDs per query
        supportive_docs = await self.pubmed.search(queries["supportive"], max_results=200)
        contradictory_docs = await self.pubmed.search(queries["contradictory"], max_results=200)

        # Merge documents preserving provenance
        merged = []
        pmids_seen = set()
        for d in (supportive_docs + contradictory_docs):
            dm = d.model_dump() if hasattr(d, "model_dump") else dict(d)
            pmid = str(dm.get("pmid"))
            if pmid in pmids_seen:
                continue
            pmids_seen.add(pmid)
            merged.append(dm)

        # BM25 fit & score
        self.bm25.fit(merged)
        bm25_scores = self.bm25.score(hypothesis)

        # Dense scores: fallback to BM25-based scores if dense unavailable
        try:
            # Attempt to compute dense using sentence-transformers if available
            from sentence_transformers import SentenceTransformer
            from sklearn.metrics.pairwise import cosine_similarity
            model = SentenceTransformer(self.config.get("dense_model", "all-MiniLM-L6-v2"))
            docs_text = [d.get("title", "") + " \n " + d.get("abstract", "") for d in merged]
            doc_emb = model.encode(docs_text, convert_to_numpy=True)
            q_emb = model.encode([hypothesis], convert_to_numpy=True)[0]
            sims = cosine_similarity([q_emb], doc_emb).flatten()
            dense_scores = {str(d.get("pmid")): float(s) for d, s in zip(merged, sims)}
            embeddings = {str(d.get("pmid")): list(vec) for d, vec in zip(merged, doc_emb)}
        except Exception:
            dense_scores = {k: 0.0 for k in bm25_scores.keys()}
            embeddings = {k: [0.0] for k in bm25_scores.keys()}

        # Build rerank initial scores (BM25 + dense proxy)
        rerank_scores = {}
        for pmid in bm25_scores.keys():
            rerank_scores[pmid] = 0.5 * bm25_scores.get(pmid, 0.0) + 0.5 * dense_scores.get(pmid, 0.0)

        # Cross-encoder rerank top documents (dominates final ordering)
        try:
            # prepare docs list (limit to top 50 by rerank_scores)
            top_candidates = sorted(rerank_scores.items(), key=lambda kv: kv[1], reverse=True)[:50]
            top_pmids = [pm for pm, _ in top_candidates]
            docs_map = {str(d.get("pmid")): d for d in merged}
            docs_for_rerank = [docs_map.get(p) for p in top_pmids if docs_map.get(p)]
            ce_scores = self.reranker.rerank(hypothesis, docs_for_rerank, top_k=50)
            # override rerank_scores for those pmids
            for pmid, s in ce_scores.items():
                rerank_scores[pmid] = s
        except Exception:
            pass

        # collect years and study types
        years = {str(d.get("pmid")): d.get("year") or d.get("publication_year") or None for d in merged}
        study_types = {str(d.get("pmid")): (d.get("publication_types") or [])[0] if (d.get("publication_types") or []) else "" for d in merged}

        final_scores = self.hybrid.compute_final_scores(bm25_scores=bm25_scores, dense_scores=dense_scores, rerank_scores=rerank_scores, years=years, study_type=study_types)

        # Select top candidates by final score
        ranked = sorted(final_scores.items(), key=lambda kv: kv[1], reverse=True)
        ranked_pmids = [pm for pm, _ in ranked]
        top_pmids = ranked_pmids[: max(100, top_k * 5)]

        # MMR for diversity
        selected = mmr_select(top_pmids, embeddings, final_scores, k=top_k)

        # Partition supportive vs contradictory using presence in initial retrievals
        supp_set = {str(d.pmid) if hasattr(d, 'pmid') else str(d.get('pmid')) for d in supportive_docs}
        contra_set = {str(d.pmid) if hasattr(d, 'pmid') else str(d.get('pmid')) for d in contradictory_docs}

        supporting_docs = [p for p in selected if p in supp_set]
        contradictory_docs = [p for p in selected if p in contra_set]

        # Ensure contradiction balancing: try to maintain 60/40 split
        desired_supp = int(top_k * 0.6)
        desired_contra = top_k - desired_supp
        final_support = supporting_docs[:desired_supp]
        final_contra = contradictory_docs[:desired_contra]

        # fill with remaining if needed
        remaining = [p for p in selected if p not in final_support + final_contra]
        for p in remaining:
            if len(final_support) < desired_supp:
                final_support.append(p)
            elif len(final_contra) < desired_contra:
                final_contra.append(p)
            else:
                break

        # Build merged ranked docs list (pmid->doc)
        pmid_to_doc = {str(d.get("pmid")): d for d in merged}
        merged_ranked_docs = [pmid_to_doc.get(p) for p in selected if pmid_to_doc.get(p)]

        # Save artifacts
        try:
            if self.debug.is_enabled():
                self.debug.save_json("retrieval/artifact_queries.json", {"queries": queries, "pmids": list(pmids_seen)})
                self.debug.save_json("retrieval/scores.json", {"bm25": bm25_scores, "dense": dense_scores, "final": final_scores})
        except Exception:
            pass

        return {
            "hypothesis": hypothesis,
            "supporting_docs": [pmid_to_doc.get(p) for p in final_support if pmid_to_doc.get(p)],
            "contradictory_docs": [pmid_to_doc.get(p) for p in final_contra if pmid_to_doc.get(p)],
            "merged_ranked_docs": merged_ranked_docs,
            "retrieval_metadata": {
                "supportive_query": queries.get("supportive"),
                "contradictory_query": queries.get("contradictory"),
                "pmids_retrieved": len(pmids_seen),
                "bm25_scores": bm25_scores,
                "dense_scores": dense_scores,
                "fused_scores": final_scores,
                "rerank_scores": rerank_scores,
            },
        }
