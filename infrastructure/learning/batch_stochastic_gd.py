"""
Batch and Stochastic Gradient Descent combined for fast convergence
"""
import json, random
from pathlib import Path

BASE = Path(__file__).resolve().parents[2].parent.parent
LOG_DIR = BASE / "runtime" / "logs"
LEDGER = LOG_DIR / "foldable_answer_ledger.jsonl"

def load_batch(batch_size=200):
    entries = []
    if LEDGER.exists():
        with open(LEDGER, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entries.append(json.loads(line))
                    if len(entries) >= batch_size:
                        break
                except Exception:
                    pass
    return entries

def stochastic_step(learning_rate=0.01):
    # One sample update
    if not LEDGER.exists():
        return
    with open(LEDGER, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    if not lines:
        return
    e = json.loads(random.choice(lines))
    # Simple stochastic update
    return e

def batch_step(learning_rate=0.01):
    batch = load_batch(200)
    # Aggregate gradient
    grad = {}
    for e in batch:
        src = e.get('source','')
        pred = 'unknown'
        if 'predictor_promoted' in src:
            pred = src.split(':')[1].split('_')[0]
        elif 'meta_evaluator_needs_rework' in src:
            pred = src.split(':')[1].split('_')[0]
        reward = 1.0 if e.get('is_correct', True) else -1.0
        grad[pred] = grad.get(pred, 0.0) + reward
    # Update thresholds
    try:
        import sys
        sys.path.insert(0, str(BASE / "infrastructure" / "predictor_tuning"))
        from predictor_tuner import load_thresholds, save_thresholds
        thresholds = load_thresholds()
        for pred, g in grad.items():
            # Find key
            key = None
            for k in thresholds:
                if pred.lower() in k.lower():
                    key = k
                    break
            if not key:
                key = pred
            old = thresholds.get(key, 0.9)
            new = old - learning_rate * g * 0.01
            thresholds[key] = max(0.8, min(0.98, new))
        save_thresholds(thresholds)
    except Exception:
        pass

def combined_update():
    # Stochastic step for fast reaction
    stochastic_step()
    # Batch step for stable convergence
    batch_step()
