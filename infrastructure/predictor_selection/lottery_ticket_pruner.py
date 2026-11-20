"""
Lottery Ticket pruning for predictor ensemble.
Keeps a sparse sub-network of weights that drive correctness.
"""
import json
from pathlib import Path
import numpy as np

BASE = Path(__file__).resolve().parents[3]
LOG_DIR = BASE / "runtime" / "logs"

def load_predictor_weights():
    # Load recent predictor stats for weight importance
    path = LOG_DIR / "predictor_stats.jsonl"
    weights = {}
    if not path.exists():
        return weights
    importances = {}
    counts = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
                pred = d.get("predictor")
                conf = float(d.get("confidence", 0.5))
                correct = 1 if d.get("correct") else 0
                # simple importance = confidence * correctness
                imp = conf * (0.5 + 0.5*correct)
                importances[pred] = importances.get(pred, 0.0) + imp
                counts[pred] = counts.get(pred, 0) + 1
            except Exception:
                pass
    for k in importances:
        weights[k] = importances[k] / max(1, counts[k])
    return weights

def prune_lottery_tickets(weights, keep_ratio=0.3, seed=42):
    if not weights:
        return [], {}
    items = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    k = max(1, int(len(items) * keep_ratio))
    winners = [name for name, _ in items[:k]]
    mask = {name: True for name, _ in items[:k]}
    mask.update({name: False for name, _ in items[k:]})
    return winners, mask

def run_lottery_ticket():
    weights = load_predictor_weights()
    winners, mask = prune_lottery_tickets(weights, keep_ratio=0.3)
    out = {
        "winners": winners,
        "mask": mask,
        "total_predictors": len(weights),
        "kept": len(winners)
    }
    out_path = LOG_DIR / "lottery_ticket_mask.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out

if __name__ == "__main__":
    print(json.dumps(run_lottery_ticket(), indent=2))
