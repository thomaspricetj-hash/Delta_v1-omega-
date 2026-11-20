"""
Bias-Variance controller with decaying flexibility and reward for approaching zero.
Starts at 3.0 and decays toward 0 over time.
Reward is given when flexibility decreases and bias/variance improves.
"""
from typing import Dict, Any
import json
from pathlib import Path

_STATE_PATH = Path(__file__).resolve().parents[3] / "runtime" / "logs" / "bias_variance_state.json"
_REWARD_LOG = Path(__file__).resolve().parents[3] / "runtime" / "logs" / "bias_variance_reward.jsonl"

def _load_state():
    if _STATE_PATH.exists():
        try:
            data = json.loads(_STATE_PATH.read_text())
            return data
        except Exception:
            pass
    return {"flexibility": 3.0, "calls": 0, "prev_flexibility": 3.0}

def _save_state(state):
    try:
        _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _STATE_PATH.write_text(json.dumps(state))
    except Exception:
        pass

def _log_reward(entry):
    try:
        _REWARD_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(_REWARD_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass

def _log_predictor_rewards(entry):
    # Propagate reward to map/graph/sharded ledger with links/index for all predictors
    try:
        from pathlib import Path
        base = Path(__file__).resolve().parents[3]
        ledger_path = base / "runtime" / "logs" / "predictor_ledger.jsonl"
        # Also write to map/graph sharded views
        map_path = base / "runtime" / "logs" / "predictor_map.json"
        graph_path = base / "runtime" / "logs" / "predictor_graph.json"
        # Build a reward payload per predictor from diagnostics
        # entry already contains global reward; we broadcast it
        reward_payload = {
            "ts": entry.get("ts"),
            "flexibility_new": entry.get("flexibility_new"),
            "reward": entry.get("reward", 0.0),
            "proximity_bonus": entry.get("proximity_bonus", 0.0),
            "source": "bias_variance"
        }
        with open(ledger_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(reward_payload) + "\n")
        # Update map/graph index files as simple append-only logs for now
        # Keep them as JSON arrays for quick inspection
        try:
            if map_path.exists():
                data = json.loads(map_path.read_text())
            else:
                data = []
            data.append(reward_payload)
            map_path.write_text(json.dumps(data))
        except Exception:
            pass
        try:
            if graph_path.exists():
                data = json.loads(graph_path.read_text())
            else:
                data = []
            data.append(reward_payload)
            graph_path.write_text(json.dumps(data))
        except Exception:
            pass
    except Exception:
        pass

def get_current_flexibility(decay_rate: float = 0.999, min_flex: float = 0.1) -> float:
    state = _load_state()
    flex = float(state.get("flexibility", 3.0))
    calls = int(state.get("calls", 0))
    prev_flex = float(state.get("prev_flexibility", flex))
    calls += 1
    # Exponential decay
    new_flex = max(min_flex, flex * decay_rate)
    state["prev_flexibility"] = flex
    state["flexibility"] = new_flex
    state["calls"] = calls
    _save_state(state)
    return new_flex

def compute_bias_variance(predictor_confidences: Dict[str, float], diagnostics: Dict[str, Any], flexibility: float = 3.0):
    import math
    confs = [v for v in predictor_confidences.values() if isinstance(v, (int, float))]
    if not confs:
        return {"bias": 0.5, "variance": 0.5, "flexibility": flexibility, "adjusted_score": 0.5, "mean_confidence": 0.5, "predictor_count": 0}
    mean_conf = sum(confs) / len(confs)
    bias = max(0.0, 1.0 - mean_conf)
    var = sum((c - mean_conf) ** 2 for c in confs) / len(confs)
    variance = math.sqrt(var)
    flexibility_factor = 1.0 + (flexibility - 1.0) * 0.1
    adjusted_score = max(0.0, min(1.0, mean_conf * flexibility_factor - bias * 0.5))
    return {
        "bias": bias,
        "variance": variance,
        "flexibility": flexibility,
        "flexibility_factor": flexibility_factor,
        "adjusted_score": adjusted_score,
        "mean_confidence": mean_conf,
        "predictor_count": len(confs)
    }

def adjust_predictor_weights(diagnostics: Dict[str, Any], flexibility: float | None = None) -> Dict[str, Any]:
    # Load state for adaptive decay
    state = _load_state()
    prev_flex = float(state.get("prev_flexibility", 3.0))
    current_flex = float(state.get("flexibility", 3.0))
    
    # Adaptive decay rate based on recent bias/variance
    # If bias is high, slow decay to allow more flexibility; if bias low, accelerate decay
    # We need current metrics to adapt, so compute a provisional metric
    predictor_confidences = {}
    for k, v in diagnostics.items():
        if k.endswith("_confidence") and isinstance(v, (int, float)):
            predictor_confidences[k] = v
    if "predictor_confidence" in diagnostics:
        predictor_confidences["predictor_confidence"] = diagnostics["predictor_confidence"]
    
    # Provisional metrics for adaptive decay
    provisional = compute_bias_variance(predictor_confidences, diagnostics, current_flex)
    bias = provisional["bias"]
    # Adaptive decay: base 0.999, increase to 0.995 if bias low, decrease to 0.9999 if bias high
    base_decay = 0.999
    adaptive_decay = max(0.99, min(0.9999, base_decay - (0.5 - bias) * 0.004))
    
    # Get new flexibility with adaptive decay
    new_flex = get_current_flexibility(decay_rate=adaptive_decay, min_flex=0.1)
    
    # Recompute metrics with new flexibility
    metrics = compute_bias_variance(predictor_confidences, diagnostics, new_flex)
    
    # Reward for getting closer to zero flexibility and improving bias/variance
    flex_delta = prev_flex - new_flex
    bias_improvement = max(0.0, provisional["bias"] - metrics["bias"])
    variance_improvement = max(0.0, provisional["variance"] - metrics["variance"])
    reward = flex_delta * 2.0 + bias_improvement * 1.5 + variance_improvement * 1.0
    # Bonus when flexibility is approaching zero
    proximity_bonus = max(0.0, (0.5 - new_flex)) * 0.5 if new_flex < 0.5 else 0.0
    reward += proximity_bonus
    
    # Log reward
    reward_entry = {
        "ts": __import__("datetime").datetime.utcnow().isoformat(),
        "flexibility_prev": prev_flex,
        "flexibility_new": new_flex,
        "flex_delta": flex_delta,
        "bias": metrics["bias"],
        "variance": metrics["variance"],
        "bias_improvement": bias_improvement,
        "reward": reward,
        "proximity_bonus": proximity_bonus
    }
    _log_reward(reward_entry)
    # Propagate reward to predictor map/graph/sharded ledger/index
    _log_predictor_rewards(reward_entry)
    
    adjustment = {
        "bias": metrics["bias"],
        "variance": metrics["variance"],
        "flexibility": metrics["flexibility"],
        "adjusted_score": metrics["adjusted_score"],
        "mean_confidence": metrics["mean_confidence"],
        "predictor_count": metrics["predictor_count"],
        "adaptive_decay": adaptive_decay,
        "reward": reward,
        "proximity_bonus": proximity_bonus
    }
    return adjustment
