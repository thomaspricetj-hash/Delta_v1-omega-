"""
Attention engine for Translation OS V1
Computes dynamic attention weights over map scores similar to transformer attention.
"""
import math
import re

def _softmax(logits):
    max_logit = max(logits)
    exps = [math.exp(l - max_logit) for l in logits]
    s = sum(exps)
    return [e/s for e in exps]

def compute_attention_weights(query, scores):
    """
    scores: dict with keys semantic, pattern, nextword, stat
    Returns dict of weights summing to 1.0
    """
    q = query.lower()
    # features
    has_math = bool(re.search(r'[0-9\+\-\*\/\^\(\)=]', q))
    is_question = '?' in q
    length = len(q.split())
    # base logits
    logits = []
    # semantic bias
    sem_logit = 0.0
    if is_question:
        sem_logit += 0.3
    if length > 5:
        sem_logit += 0.2
    logits.append(sem_logit)

    pat_logit = 0.0
    if has_math:
        pat_logit += 0.5
    if re.search(r'\bwhat is\b', q):
        pat_logit += 0.2
    logits.append(pat_logit)

    nw_logit = 0.0
    if length <= 4:
        nw_logit += 0.3
    logits.append(nw_logit)

    stat_logit = 0.0
    if length >= 3:
        stat_logit += 0.2
    logits.append(stat_logit)

    weights = _softmax(logits)
    return {
        'semantic': weights[0],
        'pattern': weights[1],
        'nextword': weights[2],
        'stat': weights[3]
    }

def apply_attention(query, scores):
    w = compute_attention_weights(query, scores)
    combined = (
        w['semantic'] * scores.get('semantic', 0.0) +
        w['pattern'] * scores.get('pattern', 0.0) +
        w['nextword'] * scores.get('nextword', 0.0) +
        w['stat'] * scores.get('stat', 0.0)
    )
    return combined, w
