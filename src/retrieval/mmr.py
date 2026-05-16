from typing import List, Dict
import numpy as np


def mmr_select(candidates: List[str], embeddings: Dict[str, List[float]], scores: Dict[str, float], k: int = 10, lambda_param: float = 0.7) -> List[str]:
    """Simple MMR selection balancing relevance (scores) and diversity (embedding cosine similarity).

    candidates: list of pmids
    embeddings: pmid -> vector
    scores: pmid -> relevance score
    """
    if not candidates:
        return []
    selected = []
    remaining = set(candidates)

    # Precompute embedding matrix
    emb_mat = {pid: np.array(vec) for pid, vec in embeddings.items()}

    # Start with top-scoring candidate
    first = max(remaining, key=lambda p: scores.get(p, 0.0))
    selected.append(first)
    remaining.remove(first)

    while len(selected) < k and remaining:
        best = None
        best_score = -1e9
        for pid in list(remaining):
            sim_to_selected = 0.0
            if selected:
                sims = [np.dot(emb_mat[pid], emb_mat[s]) / (np.linalg.norm(emb_mat[pid]) * np.linalg.norm(emb_mat[s]) + 1e-12) for s in selected if s in emb_mat]
                sim_to_selected = max(sims) if sims else 0.0
            mmr_score = lambda_param * scores.get(pid, 0.0) - (1 - lambda_param) * sim_to_selected
            if mmr_score > best_score:
                best_score = mmr_score
                best = pid
        if best is None:
            break
        selected.append(best)
        remaining.remove(best)
    return selected
