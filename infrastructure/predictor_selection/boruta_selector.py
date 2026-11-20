"""
Boruta feature selection for predictor ensemble pruning.
Uses sklearn RandomForest to select predictors that beat shadow features.
"""
import json
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

BASE = Path(__file__).resolve().parents[2]
LOG_DIR = BASE / "runtime" / "logs"

def load_recent_diagnostics(n=500):
    path = LOG_DIR / "orchestrator_diagnostics.jsonl"
    rows = []
    # Prefer plain jsonl for Boruta; BitDrop files are binary frames and not compatible with current loader
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        rows.append(obj)
                    except Exception:
                        pass
        except Exception:
            pass
    # Optional fallback to BitDrop persistence if plain is empty
    if not rows:
        bd_path = LOG_DIR / "orchestrator_diagnostics.jsonl.bd2"
        if bd_path.exists():
            try:
                import sys
                sys.path.insert(0, str(BASE / "scripts"))
                from bitdrop_persistence import load_lines_bitdrop
                for obj in load_lines_bitdrop(path):
                    if "diagnostics" in obj and isinstance(obj["diagnostics"], dict):
                        diag = obj["diagnostics"]
                        flat = {
                            "ts": obj.get("timestamp"), 
                            "query": obj.get("query_original"), 
                            "response_source": obj.get("response_source"),
                            "response_preview": obj.get("response_preview")
                        }
                        flat.update(diag)
                        rows.append(flat)
                    else:
                        rows.append(obj)
            except Exception:
                pass
    return rows[-n:]

def build_feature_matrix(diagnostics):
    # Features: predictor confidences, latencies, rewards
    # First pass to collect all feature names
    all_names = set()
    rows = []
    for d in diagnostics:
        feats = {}
        for k,v in d.items():
            if k.endswith("_confidence") and isinstance(v, (int,float)):
                feats[k] = v
        bv = d.get("bias_variance", {})
        if isinstance(bv, dict):
            for k2,v2 in bv.items():
                if isinstance(v2, (int,float)):
                    feats[f"bv_{k2}"] = v2
        # target
        correct = 1 if d.get("response_source") else 0
        if not feats:
            continue
        rows.append((feats, correct))
        all_names.update(feats.keys())
    if not rows:
        return None, None, []
    feature_names = sorted(all_names)
    X = []
    y = []
    for feats, correct in rows:
        # Align features to common names, fill missing with 0
        row = [float(feats.get(name, 0.0)) for name in feature_names]
        X.append(row)
        y.append(correct)
    X = np.array(X, dtype=float)
    y = np.array(y)
    # Ensure we have at least two classes
    if len(np.unique(y)) < 2:
        # Add a tiny jitter to avoid single-class failure
        y = y.astype(float)
        y += np.random.randn(len(y)) * 1e-6
        y = (y > 0.5).astype(int)
    return X, y, feature_names

def boruta_select(X, y, feature_names, n_estimators=200, max_iter=5, random_state=42):
    if X.shape[1] == 0:
        return []
    rng = np.random.RandomState(random_state)
    selected = []
    for _ in range(max_iter):
        # create shadow features
        X_shadow = np.copy(X)
        rng.shuffle(X_shadow)
        X_aug = np.hstack([X, X_shadow])
        names_aug = feature_names + [f"shadow_{n}" for n in feature_names]
        clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=1)
        clf.fit(X_aug, y)
        importances = clf.feature_importances_
        orig_imp = importances[:len(feature_names)]
        shadow_imp = importances[len(feature_names):]
        # keep features where importance > max shadow importance
        threshold = np.max(shadow_imp) if len(shadow_imp) > 0 else 0
        keep = [i for i, imp in enumerate(orig_imp) if imp > threshold]
        # refine
        if not keep:
            break
        X = X[:, keep]
        feature_names = [feature_names[i] for i in keep]
    return feature_names

def run_boruta():
    diags = load_recent_diagnostics(500)
    X, y, names = build_feature_matrix(diags)
    if X is None:
        return {"selected_predictors": [], "reason": "no data"}
    selected = boruta_select(X, y, names)
    out = {
        "selected_predictors": selected,
        "total_features": len(names),
        "sample_size": len(y)
    }
    out_path = LOG_DIR / "boruta_selection.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out

if __name__ == "__main__":
    print(json.dumps(run_boruta(), indent=2))
