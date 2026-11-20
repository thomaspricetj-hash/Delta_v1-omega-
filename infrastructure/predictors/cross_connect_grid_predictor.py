"""
Cross-Connecting Grid Predictor
Cross-links learning modalities, foldable maps, index/ledger, and predictor convergence
to find consensus answers via a multi-layer grid.
"""
import json
from infrastructure.predictors.predictor_wrapper import wrap_predictor
from pathlib import Path

BASE = Path(__file__).resolve().parents[2].parent.parent
LOG_DIR = BASE / "runtime" / "logs"

def _load_json(path):
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}

def _predict_impl():
    q = (query or "").lower().strip()
    # Simple cross-grid heuristic: look for consensus in predictor_convergence and predictor_map
    conv_path = LOG_DIR / "predictor_convergence.jsonl"
    map_path = LOG_DIR / "predictor_map.json"
    consensus = []
    try:
        # Load recent convergences for this query
        recent = []
        if conv_path.exists():
            with open(conv_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        e = json.loads(line)
                        if e.get("qcanon", "").lower() == q or q in e.get("qcanon","").lower():
                            recent.append(e)
                    except Exception:
                        pass
        # Map hits
        pmap = _load_json(map_path)
        hits = pmap.get(q, {})
    except Exception:
        recent = []
        hits = {}
    
    # If we have multiple predictor signals, build consensus
    if recent:
        scores = [e.get("convergence_score",0) for e in recent[-10:]]
        avg_score = sum(scores)/len(scores) if scores else 0.0
        if avg_score > 0.7:
            # Return a placeholder consensus answer from diagnostics or map
            # Fallback to most common response in recent logs
            # For now, return a synthetic consensus marker
            return {
                "response": f"[cross-grid consensus] {query}",
                "confidence": avg_score,
                "source": "cross_connect_grid",
                "hits": len(recent)
            }
    # No strong consensus
    return None

predict = wrap_predictor(_predict_impl, "cross_connect_grid_predictor")
