"""
Mini-batch gradient descent for predictor weight adjustment
"""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[2].parent.parent
LOG_DIR = BASE / "runtime" / "logs"
LEDGER = LOG_DIR / "foldable_answer_ledger.jsonl"

def load_batch(batch_size=100):
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

def update_weights(learning_rate=0.01):
    batch = load_batch(100)
    # Simple gradient: reward correct, penalize wrong
    grads = {}
    for e in batch:
        src = e.get('source','')
        is_correct = e.get('is_correct', True)
        # Extract predictor name
        pred = 'unknown'
        if 'predictor_promoted' in src:
            pred = src.split(':')[1].split('_')[0]
        elif 'meta_evaluator_needs_rework' in src:
            pred = src.split(':')[1].split('_')[0]
        # Gradient step
        reward = 1.0 if is_correct else -1.0
        grads[pred] = grads.get(pred, 0.0) + reward
    # Apply simple update to predictor tuner thresholds
    try:
        sys_path = Path(__file__).resolve().parent.parent.parent
        import sys
        sys.path.insert(0, str(sys_path / "infrastructure" / "predictor_tuning"))
        from predictor_tuner import load_thresholds, save_thresholds
        thresholds = load_thresholds()
        for pred, g in grads.items():
            # Find matching key
            key = None
            for k in thresholds:
                if pred.lower() in k.lower():
                    key = k
                    break
            if not key:
                key = pred
            # Simple gradient descent on threshold
            old = thresholds.get(key, 0.9)
            # Higher reward -> lower threshold, lower reward -> higher threshold
            new = old - learning_rate * g * 0.01
            thresholds[key] = max(0.8, min(0.98, new))
        save_thresholds(thresholds)
    except Exception:
        pass
