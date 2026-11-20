"""
Multi-head attention over memory graph using native embeddings.
"""
import numpy as np
from infrastructure.embeddings.embeddings import store as emb_store
from infrastructure.master_map.master_map import get_master_map

NUM_HEADS = 4
HEAD_DIM = 64

def _split_heads(vec, num_heads):
    # vec is 256-dim, split into heads
    parts = np.array_split(vec, num_heads)
    return parts

def attend_multihead(query, top_k=8):
    q_vec = emb_store.get(query)
    heads = _split_heads(q_vec, NUM_HEADS)
    mm = get_master_map()
    keys = list(mm.keys())[:2000]
    scores = []
    for k in keys:
        v = emb_store.get(k)
        v_heads = _split_heads(v, NUM_HEADS)
        # Average head similarity
        sims = [float(np.dot(hq, hv)) for hq, hv in zip(heads, v_heads)]
        score = sum(sims) / NUM_HEADS
        scores.append((k, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]
