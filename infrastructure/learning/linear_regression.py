"""
Linear Regression with Normal Equation for predictor calibration
"""
import json, numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parents[2].parent.parent
LOG_DIR = BASE / "runtime" / "logs"
LEDGER = LOG_DIR / "foldable_answer_ledger.jsonl"

def load_features(batch_size=200):
    X = []
    y = []
    if LEDGER.exists():
        with open(LEDGER, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    e = json.loads(line)
                    # Features: confidence, support, oppose
                    conf = e.get('confidence', 0.5)
                    # Simple feature vector
                    X.append([1.0, conf])
                    y.append(1.0 if e.get('is_correct', True) else 0.0)
                    if len(X) >= batch_size:
                        break
                except Exception:
                    pass
    return np.array(X), np.array(y)

def normal_equation(X, y):
    # Add intercept already in X
    XtX = X.T @ X
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(XtX)
    theta = XtX_inv @ X.T @ y
    return theta

def calibrate_predictors():
    X, y = load_features(200)
    if len(X) < 10:
        return
    theta = normal_equation(X, y)
    # theta[0] = intercept, theta[1] = weight for confidence
    # Use theta to adjust thresholds
    try:
        import sys
        sys.path.insert(0, str(BASE / "infrastructure" / "predictor_tuning"))
        from predictor_tuner import load_thresholds, save_thresholds
        thresholds = load_thresholds()
        # Simple calibration: shift thresholds by theta
        for k in thresholds:
            # Adjust toward better calibration
            thresholds[k] = max(0.8, min(0.98, thresholds[k] + 0.01 * theta[1]))
        save_thresholds(thresholds)
    except Exception:
        pass
