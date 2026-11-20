"""
Attention over own memory graph using native embeddings.
"""
import numpy as np
from infrastructure.embeddings.embeddings import store as emb_store
from infrastructure.master_map.master_map import get_master_map

def attend(query: str, top_k: int = 8):
    """
    Simple cosine attention over master map keys.
    Returns list of (key, score) sorted by similarity.
    """
    q_vec = emb_store.get(query)
    candidates = []
    # Use master map keys as memory nodes
    mm = get_master_map()
    # Limit scan for speed
    keys = list(mm.keys())[:5000]
    # Use cached matmul for cheap CPU attention
    try:
        from infrastructure.predictors.matrix_multiply_predictor import cached_matmul
    except Exception:
        cached_matmul = None
    for k in keys:
        try:
            v = emb_store.get(k)
            if cached_matmul:
                dot = cached_matmul(q_vec, v, op="*")
                score = float(np.dot(q_vec, v) if isinstance(dot, (list,)) else dot if dot is not None else np.dot(q_vec, v))
            else:
                score = float(np.dot(q_vec, v))
            candidates.append((k, score))
        except Exception:
            continue
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[:top_k]

def attend_with_context(query: str, context: list[str], top_k: int = 8):
    """
    Attend with query + context bias.
    """
    # Average embedding of context
    if context:
        ctx_vecs = [emb_store.get(c) for c in context]
        ctx_vec = np.mean(ctx_vecs, axis=0)
        q_vec = emb_store.get(query)
        q_vec = 0.7 * q_vec + 0.3 * ctx_vec
        # Normalize
        q_vec = q_vec / (np.linalg.norm(q_vec) + 1e-9)
    else:
        q_vec = emb_store.get(query)

    mm = get_master_map()
    keys = list(mm.keys())[:5000]
    scores = []
    try:
        from infrastructure.predictors.matrix_multiply_predictor import cached_matmul
    except Exception:
        cached_matmul = None
    for k in keys:
        v = emb_store.get(k)
        if cached_matmul:
            dot = cached_matmul(q_vec, v, op="*")
            score = float(np.dot(q_vec, v) if isinstance(dot, (list,)) else dot if dot is not None else np.dot(q_vec, v))
        else:
            score = float(np.dot(q_vec, v))
        scores.append((k, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]
