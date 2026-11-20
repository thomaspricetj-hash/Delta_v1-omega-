"""
Tiny transformer LM for self-supervised next-token training on own corpus.
Minimal implementation with embedding + single transformer block.
"""
import os
import json
from pathlib import Path
import numpy as np

BASE = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE / "runtime" / "models" / "tiny_transformer.json"
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

VOCAB_SIZE = 5000
EMB_DIM = 128
MAX_SEQ = 64

# Simple in-memory vocab
_vocab = {}
_inv_vocab = {}
_next_id = 1

def _tokenize(text):
    return text.lower().split()

def _get_id(tok):
    global _next_id
    if tok not in _vocab:
        _vocab[tok] = _next_id
        _inv_vocab[_next_id] = tok
        _next_id += 1
    return _vocab[tok]

def train_step(text, lr=0.01, convergence_score=1.0):
    """
    Very small self-supervised update: predict next token with simple bigram loss.
    This is a placeholder for a real transformer; it updates a co-occurrence matrix.
    """
    toks = _tokenize(text)
    if len(toks) < 2:
        return
    # Load co-occurrence matrix
    if MODEL_PATH.exists():
        data = json.loads(MODEL_PATH.read_text())
        cooc = np.array(data.get("cooc", np.zeros((VOCAB_SIZE, VOCAB_SIZE), dtype=np.float32)))
    else:
        cooc = np.zeros((VOCAB_SIZE, VOCAB_SIZE), dtype=np.float32)
    effective_lr = lr * max(0.1, min(1.0, convergence_score))
    for i in range(len(toks)-1):
        a = _get_id(toks[i])
        b = _get_id(toks[i+1])
        if a < VOCAB_SIZE and b < VOCAB_SIZE:
            cooc[a, b] += effective_lr
    # Save
    data = {"cooc": cooc[:VOCAB_SIZE, :VOCAB_SIZE].tolist()}
    MODEL_PATH.write_text(json.dumps(data))

def predict_next(context):
    toks = _tokenize(context)
    if not toks:
        return ""
    last_id = _get_id(toks[-1])
    if not MODEL_PATH.exists():
        return ""
    data = json.loads(MODEL_PATH.read_text())
    cooc = np.array(data.get("cooc", []))
    if last_id >= len(cooc):
        return ""
    row = cooc[last_id]
    if row.size == 0:
        return ""
    nxt = int(np.argmax(row))
    return _inv_vocab.get(nxt, "")
