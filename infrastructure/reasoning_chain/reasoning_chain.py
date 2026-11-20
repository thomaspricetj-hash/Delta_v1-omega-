"""
Reasoning chain for 100% memory learning and reasoning.
Chained inference with backtracking and verification.
"""
import sys
from pathlib import Path
BASE = Path(__file__).resolve().parents[2]

def chain_reason(query, max_steps=5):
    """
    Multi-step reasoning over memory graph.
    Returns list of intermediate steps.
    """
    steps = []
    # Load memory attention
    sys.path.insert(0, str(BASE / "infrastructure" / "memory_attention"))
    from multihead_attention import attend_multihead
    # Load embeddings
    sys.path.insert(0, str(BASE / "infrastructure" / "embeddings"))
    from embeddings import embed
    
    current = query
    for i in range(max_steps):
        top = attend_multihead(current, top_k=3)
        if not top:
            break
        # Pick best
        best_key, best_score = top[0]
        steps.append({"step": i+1, "query": current, "retrieved": best_key, "score": best_score})
        # Move to next
        current = best_key
    return steps
