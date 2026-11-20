"""
Auto-tuning for predictor ensemble thresholds
"""
import json
from pathlib import Path

STATS_PATH = Path("runtime/logs/predictor_stats.jsonl")
TUNING_PATH = Path("runtime/logs/predictor_tuning.jsonl")
THRESHOLDS_PATH = Path("runtime/logs/predictor_thresholds.json")

DEFAULT_THRESHOLDS = {
    "rulebook": 1.0,
    "context_cache": 0.9,
    "semantic_similarity": 0.92,
    "micro_cluster": 0.85,
    "next_token": 0.88,
    "analogical": 0.8
}

def load_thresholds():
    if THRESHOLDS_PATH.exists():
        try:
            with open(THRESHOLDS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return DEFAULT_THRESHOLDS.copy()

def save_thresholds(thresh):
    try:
        THRESHOLDS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(THRESHOLDS_PATH, "w", encoding="utf-8") as f:
            json.dump(thresh, f)
    except Exception:
        pass

def log_stats(predictor_name, confidence, correct, latency_ms):
    try:
        entry = {
            "predictor": predictor_name,
            "confidence": confidence,
            "correct": bool(correct),
            "latency_ms": latency_ms
        }
        with open(STATS_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass

def auto_tune():
    # Simple hill-climb based on recent stats
    thresholds = load_thresholds()
    try:
        # Read last 1000 stats
        stats = []
        if STATS_PATH.exists():
            with open(STATS_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        stats.append(json.loads(line))
                    except Exception:
                        pass
        # Keep last 1000
        stats = stats[-1000:]
        if not stats:
            return thresholds
        
        # Group by predictor
        from collections import defaultdict
        per_pred = defaultdict(list)
        for s in stats:
            per_pred[s["predictor"]].append(s)
        
        for pred, items in per_pred.items():
            if not items:
                continue
            precision = sum(1 for i in items if i["correct"]) / len(items)
            hit_rate = len(items) / max(1, len(stats))
            avg_latency = sum(i["latency_ms"] for i in items) / len(items)
            
            # Adjust thresholds
            if precision < 0.85:
                thresholds[pred] = min(0.99, thresholds.get(pred, 0.9) + 0.01)
            elif precision > 0.95 and hit_rate < 0.3:
                thresholds[pred] = max(0.7, thresholds.get(pred, 0.9) - 0.01)
            
            # Log tuning
            try:
                with open(TUNING_PATH, "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "predictor": pred,
                        "precision": precision,
                        "hit_rate": hit_rate,
                        "avg_latency_ms": avg_latency,
                        "threshold": thresholds.get(pred)
                    }) + "\n")
            except Exception:
                pass
    except Exception:
        pass
    
    save_thresholds(thresholds)
    return thresholds
