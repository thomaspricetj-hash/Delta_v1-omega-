"""
Self-supervised next-token training on own corpus.
Minimal implementation using incremental embedding updates with contrastive loss.
"""
import json
import sys
from pathlib import Path
import numpy as np
BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / "infrastructure"))
from embeddings.embeddings import embed, update_embedding

TRAIN_LOG = BASE / "runtime" / "logs" / "self_supervised.jsonl"
TRAIN_LOG.parent.mkdir(parents=True, exist_ok=True)

def _tokenize(text: str):
    # Very simple whitespace tokenization
    return text.strip().split()

def train_on_pair(query: str, answer: str, lr: float = 0.01, reward: float = 1.0):
    """
    Self-supervised update: bring query embedding closer to answer embedding,
    push away from random negatives.
    reward >0 pushes toward correct, reward <0 punishes.
    """
    q_tokens = _tokenize(query)
    a_tokens = _tokenize(answer)

    # Update query-answer similarity
    for qt in q_tokens:
        for at in a_tokens:
            q_vec = embed(qt)
            a_vec = embed(at)
            # Contrastive pull scaled by reward
            delta = lr * reward * (a_vec - q_vec)
            update_embedding(qt, delta * 0.5)
            update_embedding(at, -delta * 0.5)

    # Log convergence map reward
    try:
        sys_path = __import__('sys').path
        sys_path.insert(0, str(Path(__file__).resolve().parents[2] / "infrastructure" / "convergence_map"))
        from convergence_map import add_reward_signal
        add_reward_signal(query, answer, reward)
    except Exception:
        pass

    # Log
    entry = {
        "type": "self_supervised_update",
        "query": query,
        "answer": answer,
        "tokens_q": len(q_tokens),
        "tokens_a": len(a_tokens)
    }
    with TRAIN_LOG.open('a', encoding='utf-8') as f:
        f.write(json.dumps(entry) + "\n")

def train_stream(pairs, lr=0.01):
    for q,a in pairs:
        train_on_pair(q, a, lr)
