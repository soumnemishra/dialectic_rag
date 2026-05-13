"""
Dense Retriever using sentence-transformers for semantic search.

Provides embedding-based retrieval to complement BM25 keyword search.
"""

from typing import Any, List, Tuple, Optional
import logging

from src.config import settings

logger = logging.getLogger(__name__)


_MODEL_CACHE: dict[str, dict[str, Any]] = {}


def _best_device() -> str:
    """Select the fastest available torch device for local embedding inference."""
    try:
        import torch
    except Exception:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _get_cached_sentence_transformer(model_name: str, device: str) -> Any:
    """Load a SentenceTransformer once per model/device combination."""
    device_cache = _MODEL_CACHE.setdefault(model_name, {})
    cached_model = device_cache.get(device)
    if cached_model is not None:
        return cached_model

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.warning("sentence-transformers not installed. Dense retrieval disabled.")
        device_cache[device] = False
        return False

    model = SentenceTransformer(model_name, device=device)
    device_cache[device] = model
    logger.info("SentenceTransformer cached: %s on %s", model_name, device)
    return model


class DenseRetriever:
    """
    Dense retriever using sentence-transformers for semantic search.
    
    Encodes queries and documents into embeddings and computes similarity.
    """
    
    # BioBERT-like model for medical text (or general purpose fallback)
    DEFAULT_MODEL = "all-MiniLM-L6-v2"  # Fast, good quality
    MEDICAL_MODEL = "pritamdeka/S-PubMedBert-MS-MARCO"  # Medical-specific
    
    def __init__(self, model_name: str = None, use_medical: bool = False, model: Any | None = None):
        """
        Initialize dense retriever.
        
        Args:
            model_name: HuggingFace model name for embeddings.
            use_medical: Use medical-specific model (slower but better for medical text).
        """
        if model_name:
            self.model_name = model_name
        elif use_medical:
            self.model_name = self.MEDICAL_MODEL
        else:
            self.model_name = self.DEFAULT_MODEL
        
        self._device = _best_device()
        self._model = model if model is not None else _get_cached_sentence_transformer(self.model_name, self._device)
        logger.info(f"DenseRetriever initialized (model: {self.model_name}, device={self._device})")
    
    def _load_model(self):
        """Return the shared sentence transformer model."""
        return self._model
    
    def encode_query(self, query: str) -> Optional[List[float]]:
        """Encode a query into an embedding vector."""
        model = self._load_model()
        if not model:
            return None
        return model.encode(query, convert_to_numpy=True).tolist()
    
    def encode_documents(self, documents: List[str]) -> Optional[List[List[float]]]:
        """Encode multiple documents into embedding vectors."""
        model = self._load_model()
        if not model:
            return None
        return model.encode(documents, convert_to_numpy=True).tolist()
    
    def compute_similarity(
        self, 
        query_embedding: List[float], 
        doc_embeddings: List[List[float]]
    ) -> List[float]:
        """Compute cosine similarity between query and documents."""
        import numpy as np
        
        query_vec = np.array(query_embedding)
        doc_vecs = np.array(doc_embeddings)
        
        # Normalize vectors
        query_norm = query_vec / np.linalg.norm(query_vec)
        doc_norms = doc_vecs / np.linalg.norm(doc_vecs, axis=1, keepdims=True)
        
        # Cosine similarity
        similarities = np.dot(doc_norms, query_norm)
        return similarities.tolist()
    
    def rank_documents(
        self, 
        query: str, 
        documents: List[str], 
        doc_ids: List[str]
    ) -> List[Tuple[str, str, float]]:
        """
        Rank documents by semantic similarity to query.
        
        Returns:
            List of (doc_id, doc_text, similarity_score) tuples, sorted by score.
        """
        query_emb = self.encode_query(query)
        if query_emb is None:
            return [(doc_id, doc, 0.0) for doc_id, doc in zip(doc_ids, documents)]
        
        doc_embs = self.encode_documents(documents)
        if doc_embs is None:
            return [(doc_id, doc, 0.0) for doc_id, doc in zip(doc_ids, documents)]
        
        scores = self.compute_similarity(query_emb, doc_embs)
        
        # Combine and sort
        results = list(zip(doc_ids, documents, scores))
        results.sort(key=lambda x: x[2], reverse=True)
        
        return results


def reciprocal_rank_fusion(
    ranked_lists: List[List[Tuple[str, float]]], 
    k: int = 60
) -> List[Tuple[str, float]]:
    """
    Reciprocal Rank Fusion (RRF) to combine multiple ranked lists.
    
    RRF score = sum(1 / (k + rank)) for each list where item appears.
    
    Args:
        ranked_lists: List of ranked lists, each containing (doc_id, score) tuples.
        k: RRF constant (default 60, as per original paper).
    
    Returns:
        Fused list of (doc_id, fused_score) tuples, sorted by score.
    """
    fusion_scores = {}
    
    for ranked_list in ranked_lists:
        for rank, (doc_id, _) in enumerate(ranked_list, start=1):
            if doc_id not in fusion_scores:
                fusion_scores[doc_id] = 0.0
            fusion_scores[doc_id] += 1.0 / (k + rank)
    
    # Sort by fused score
    fused = [(doc_id, score) for doc_id, score in fusion_scores.items()]
    fused.sort(key=lambda x: x[1], reverse=True)
    
    logger.info(f"RRF fusion: {len(ranked_lists)} lists -> {len(fused)} unique docs")
    return fused


# =============================================================================
# MedCPT Retriever (State-of-the-Art for Biomedical)
# =============================================================================

class MedCPTRetriever:
    """
    MedCPT (Contrastive Pre-trained Transformers) for biomedical retrieval.
    
    Trained on 255 million query-article pairs from PubMed search logs.
    Uses bi-encoder for fast initial retrieval and cross-encoder for reranking.
    
    Models:
        - Query Encoder: ncbi/MedCPT-Query-Encoder
        - Article Encoder: ncbi/MedCPT-Article-Encoder (for bi-encoder mode)
        - Cross-Encoder: ncbi/MedCPT-Cross-Encoder (for reranking)
    """
    
    QUERY_ENCODER = "ncbi/MedCPT-Query-Encoder"
    ARTICLE_ENCODER = "ncbi/MedCPT-Article-Encoder"
    CROSS_ENCODER = "ncbi/MedCPT-Cross-Encoder"
    
    def __init__(self, use_cross_encoder: bool = True):
        """
        Initialize MedCPT retriever.
        
        Args:
            use_cross_encoder: Use cross-encoder for reranking (more accurate but slower).
        """
        self.use_cross_encoder = use_cross_encoder
        self._query_encoder = None
        self._article_encoder = None
        self._cross_encoder = None
        self._tokenizer = None
        logger.info(f"MedCPTRetriever initialized (cross_encoder={use_cross_encoder})")

    def _get_device(self) -> str:
        import torch
        if not torch.cuda.is_available():
            return "cpu"
        # No local Ollama models expected in Gemini-only deployments;
        # prefer GPU when available.
        return "cuda"
    
    def _load_encoders(self):
        """Lazy load MedCPT encoders."""
        if self._query_encoder is None:
            try:
                from transformers import AutoTokenizer, AutoModel
                import torch
                
                self._tokenizer = AutoTokenizer.from_pretrained(self.QUERY_ENCODER)
                self._query_encoder = AutoModel.from_pretrained(self.QUERY_ENCODER)
                self._article_encoder = AutoModel.from_pretrained(self.ARTICLE_ENCODER)
                
                # Move to GPU if available
                device = self._get_device()
                self._query_encoder = self._query_encoder.to(device)
                self._article_encoder = self._article_encoder.to(device)
                self._device = device
                
                logger.info(f"MedCPT encoders loaded on {device}")
            except Exception as e:
                logger.warning(f"Failed to load MedCPT encoders: {e}")
                self._query_encoder = False
        return self._query_encoder
    
    def _load_cross_encoder(self):
        """Lazy load MedCPT cross-encoder."""
        if self._cross_encoder is None:
            try:
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
                import torch
                
                if self._tokenizer is None:
                    self._tokenizer = AutoTokenizer.from_pretrained(self.CROSS_ENCODER)
                
                self._cross_encoder = AutoModelForSequenceClassification.from_pretrained(
                    self.CROSS_ENCODER
                )
                
                device = self._get_device()
                self._cross_encoder = self._cross_encoder.to(device)
                self._device = device
                
                logger.info(f"MedCPT cross-encoder loaded on {device}")
            except Exception as e:
                logger.warning(f"Failed to load MedCPT cross-encoder: {e}")
                self._cross_encoder = False
        return self._cross_encoder
    
    def encode_query(self, query: str) -> Optional[List[float]]:
        """Encode query using MedCPT query encoder."""
        import torch
        
        encoder = self._load_encoders()
        if not encoder:
            return None
        
        with torch.no_grad():
            inputs = self._tokenizer(
                query, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512
            ).to(self._device)
            
            outputs = self._query_encoder(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
        
        return embedding.tolist()
    
    def encode_articles(self, articles: List[str]) -> Optional[List[List[float]]]:
        """Encode articles using MedCPT article encoder."""
        import torch
        
        encoder = self._load_encoders()
        if not encoder:
            return None
        
        with torch.no_grad():
            inputs = self._tokenizer(
                articles,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self._device)
            outputs = self._article_encoder(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy().tolist()

        return embeddings
    
    def rerank(
        self, 
        query: str, 
        documents: List[str], 
        doc_ids: List[str],
        top_k: int = 10
    ) -> List[Tuple[str, str, float]]:
        """
        Rerank documents using MedCPT cross-encoder.
        
        Returns:
            List of (doc_id, doc_text, score) tuples, sorted by score.
        """
        import torch
        
        cross_encoder = self._load_cross_encoder()
        if not cross_encoder:
            # Fallback: return original order
            return [(doc_id, doc, 0.0) for doc_id, doc in zip(doc_ids, documents)]
        
        with torch.no_grad():
            queries = [query] * len(documents)
            inputs = self._tokenizer(
                queries,
                documents,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self._device)

            outputs = self._cross_encoder(**inputs)
            scores_array = outputs.logits.squeeze(-1).cpu().numpy()
            if scores_array.ndim == 0:
                scores = [float(scores_array)]
            else:
                scores = scores_array.tolist()
        
        # Combine and sort
        results = list(zip(doc_ids, documents, scores))
        results.sort(key=lambda x: x[2], reverse=True)
        
        logger.info(f"MedCPT reranked {len(documents)} docs. Top score: {results[0][2]:.3f}")
        
        return results[:top_k]
    
    def rank_documents(
        self, 
        query: str, 
        documents: List[str], 
        doc_ids: List[str]
    ) -> List[Tuple[str, str, float]]:
        """
        Rank documents using MedCPT.
        
        Uses cross-encoder if enabled, otherwise bi-encoder similarity.
        """
        if self.use_cross_encoder:
            return self.rerank(query, documents, doc_ids)
        else:
            # Bi-encoder mode
            query_emb = self.encode_query(query)
            if query_emb is None:
                return [(doc_id, doc, 0.0) for doc_id, doc in zip(doc_ids, documents)]
            
            doc_embs = self.encode_articles(documents)
            if doc_embs is None:
                return [(doc_id, doc, 0.0) for doc_id, doc in zip(doc_ids, documents)]
            
            import numpy as np
            query_vec = np.array(query_emb)
            doc_vecs = np.array(doc_embs)
            
            # Normalize and compute similarity
            query_norm = query_vec / np.linalg.norm(query_vec)
            doc_norms = doc_vecs / np.linalg.norm(doc_vecs, axis=1, keepdims=True)
            scores = np.dot(doc_norms, query_norm).tolist()
            
            results = list(zip(doc_ids, documents, scores))
            results.sort(key=lambda x: x[2], reverse=True)
            
            return results

