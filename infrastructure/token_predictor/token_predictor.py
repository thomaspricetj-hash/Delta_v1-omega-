import json
from collections import defaultdict, Counter
from pathlib import Path
import math
from infrastructure.predictors.predictor_base import PredictorCache, log_predictor_event, log_reward, get_helpers

BASE = Path(__file__).resolve().parent.parent.parent
STORE_DIR = BASE / 'runtime' / 'logs' / 'token_predictor_shards'
STORE_DIR.mkdir(parents=True, exist_ok=True)
STORE = STORE_DIR / 'token_predictor.json'
INDEX_PATH = STORE_DIR / 'token_predictor_index.json'
LEDGER_PATH = STORE_DIR / 'token_predictor_ledger.jsonl'

transitions = defaultdict(Counter)
_MAX_RAM_ITEMS = 10000
_ACCESS_ORDER = []
_BATCH_BUFFER = []
_BATCH_SIZE = 200

def load():
    global transitions
    if STORE.exists():
        try:
            data = json.load(open(STORE, 'r', encoding='utf-8'))
            for k,v in data.items():
                transitions[k].update(v)
        except Exception:
            pass

load()

# Production-grade cache, helpers, and reward logging
_predictor_cache = PredictorCache("token_predictor")
_helpers = get_helpers()

def train_step(prev_tokens, next_token):
    if len(prev_tokens) < 2 or not next_token:
        return
    # Optional decomposition for long contexts
    chunks = _helpers["decompose_problem"](str(prev_tokens))
    for chunk in chunks:
        tokens = chunk.split()
        for i in range(len(tokens)-1):
            key = ' '.join(tokens[i:i+2])
            transitions[key][next_token] += 1
    log_predictor_event("token_predictor", "train_step", {"prev_len": len(prev_tokens), "next_token": str(next_token)})

def predict(context_tokens, top_k=3):
    if len(context_tokens) < 2:
        return []
    key = ' '.join(context_tokens[-2:])
    # Cache lookup with Bloom filter
    cached_val, hit = _predictor_cache.get(key)
    if hit:
        log_predictor_event("token_predictor", "cache_hit", {"key": key})
        return cached_val
    if key not in transitions:
        log_predictor_event("token_predictor", "miss", {"key": key})
        return []
    cnt = transitions[key]
    total = sum(cnt.values())
    results = []
    for t,_ in cnt.most_common(top_k):
        c = cnt[t]
        prob = c / total if total > 0 else 0
        log_prob = math.log(prob) if prob > 0 else float('-inf')
        results.append({'token': t, 'log_prob': log_prob, 'prob': prob})
    # Cache result
    _predictor_cache.set(key, results)
    # Confidence estimation and reward logging
    confidence = _helpers["estimate_confidence"](hit=True)
    reward = 0.5 + 0.5 * max(r['prob'] for r in results) if results else 0.1
    log_reward("token_predictor", reward, confidence, query=str(context_tokens))
    log_predictor_event("token_predictor", "predict", {"key": key, "confidence": confidence, "reward": reward})
    return results

def save():
    try:
        STORE.parent.mkdir(parents=True, exist_ok=True)
        with open(STORE,'w', encoding='utf-8') as f:
            json.dump({k:dict(v) for k,v in transitions.items()}, f)
    except Exception:
        pass
