"""
Vector-based attention for Translation OS V1
Implements the classic Attention(Q,K,V) = softmax(QK^T / sqrt(d)) V
with multi-head style using simple hash embeddings.
"""
import math
import re
import hashlib
from collections import defaultdict

EMBED_DIM = 64
NUM_HEADS = 4

def _hash_vec(token, dim=EMBED_DIM):
    # deterministic pseudo embedding from token hash
    h = hashlib.sha256(token.encode()).digest()
    vec = []
    for i in range(dim):
        b = h[i % len(h)]
        vec.append((b / 255.0) * 2 - 1)
    return vec

def _mat_vec_mul(mat, vec):
    # mat is list of rows
    res = []
    for row in mat:
        s = sum(r*v for r,v in zip(row, vec))
        res.append(s)
    return res

def _softmax(scores):
    max_s = max(scores)
    exps = [math.exp(s - max_s) for s in scores]
    sum_exps = sum(exps) or 1.0
    return [e / sum_exps for e in exps]

def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

class SimpleAttention:
    def __init__(self, dim=EMBED_DIM, heads=NUM_HEADS):
        self.dim = dim
        self.heads = heads
        # random projection matrices per head for Q,K,V – deterministic from seed
        self.Wq = []
        self.Wk = []
        self.Wv = []
        for h in range(heads):
            # simple random matrices using hash of head id
            seed = f"head{h}"
            self.Wq.append([_hash_vec(f"{seed}_q{i}", dim) for i in range(dim)])
            self.Wk.append([_hash_vec(f"{seed}_k{i}", dim) for i in range(dim)])
            self.Wv.append([_hash_vec(f"{seed}_v{i}", dim) for i in range(dim)])

    def project(self, W, x):
        # W is list of rows, x is vector
        return _mat_vec_mul(W, x)

    def attend(self, tokens):
        # Build embeddings for each token
        embeds = [_hash_vec(t) for t in tokens]
        n = len(embeds)
        if n == 0:
            return []
        outputs = []
        for h in range(self.heads):
            Q = [self.project(self.Wq[h], e) for e in embeds]
            K = [self.project(self.Wk[h], e) for e in embeds]
            V = [self.project(self.Wv[h], e) for e in embeds]
            # Compute attention for each query token
            head_out = []
            for i in range(n):
                scores = []
                for j in range(n):
                    # dot product Q_i . K_j
                    dot = sum(q*k for q,k in zip(Q[i], K[j]))
                    scores.append(dot / math.sqrt(self.dim))
                weights = _softmax(scores)
                # weighted sum of V
                out_vec = [0.0]*self.dim
                for j in range(n):
                    w = weights[j]
                    for d in range(self.dim):
                        out_vec[d] += w * V[j][d]
                head_out.append(out_vec)
            outputs.append(head_out)
        # average heads
        avg_out = []
        for i in range(n):
            vec = [0.0]*self.dim
            for h in range(self.heads):
                for d in range(self.dim):
                    vec[d] += outputs[h][i][d] / self.heads
            avg_out.append(vec)
        return avg_out

def attention_for_query(query):
    tokens = tokenize(query)
    att = SimpleAttention()
    vectors = att.attend(tokens)
    # Return attention weights matrix for inspection
    # For simplicity return token -> weighted context summary
    return {
        'tokens': tokens,
        'embed_dim': EMBED_DIM,
        'heads': NUM_HEADS,
        'output_vectors': vectors
    }
