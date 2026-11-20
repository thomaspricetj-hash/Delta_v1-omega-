"""
Least Squares routine for predictor error minimization
"""
import json, numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parents[2].parent.parent
LOG_DIR = BASE / "runtime" / "logs"
LEDGER = LOG_DIR / "foldable_answer_ledger.jsonl"

def load_data(batch_size=300):
    X = []
    y = []
    if LEDGER.exists():
        with open(LEDGER, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    e = json.loads(line)
                    conf = e.get('confidence', 0.5)
                    # Features: confidence, support, oppose
                    sup = 1 if e.get('is_correct', True) else 0
                    X.append([1.0, conf, sup])
                    y.append(1.0 if e.get('is_correct', True) else 0.0)
                    if len(X) >= batch_size:
                        break
                except Exception:
                    pass
    return np.array(X), np.array(y)

def least_squares_fit():
    X, y = load_data(300)
    if len(X) < 10:
        return None
    # Solve min ||X theta - y||^2
    theta, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
    return theta

def apply_least_squares():
    theta = least_squares_fit()
    if theta is None:
        return
    # Use theta to adjust predictor thresholds
    try:
        import sys
        sys.path.insert(0, str(BASE / "infrastructure" / "predictor_tuning"))
        from predictor_tuner import load_thresholds, save_thresholds
        thresholds = load_thresholds()
        # Simple adjustment based on confidence weight
        w_conf = theta[1] if len(theta) > 1 else 0
        for k in thresholds:
            thresholds[k] = max(0.8, min(0.98, thresholds[k] + 0.005 * w_conf))
        save_thresholds(thresholds)
    except Exception:
        pass
