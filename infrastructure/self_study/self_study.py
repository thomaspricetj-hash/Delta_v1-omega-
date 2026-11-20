"""
Self-study module: ability to study its own thinking
Logs reasoning traces, confidence, emergence and builds reflection index
"""
import json
from pathlib import Path
from datetime import datetime

LOG_PATH = Path("runtime/logs/self_study.jsonl")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

THOUGHT_INDEX_PATH = Path("runtime/logs/thought_index.json")
if THOUGHT_INDEX_PATH.exists():
    try:
        with open(THOUGHT_INDEX_PATH, "r", encoding="utf-8") as f:
            THOUGHT_INDEX = json.load(f)
    except Exception:
        THOUGHT_INDEX = {}
else:
    THOUGHT_INDEX = {}

def _save_log(entry):
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

def study_thinking(query, diagnostics):
    """Study own thinking for a query and update reflection index"""
    try:
        reasoning_thoughts = diagnostics.get("reasoning_thoughts", [])
        confidence_prior = diagnostics.get("confidence_prior", 0.5)
        fiber_harmony_score = diagnostics.get("fiber_harmony_score", 0.5)
        emergence_score = diagnostics.get("emergence_intersection_score", 0.0)
        reward_push = diagnostics.get("reward_push_applied", False)
        response_source = diagnostics.get("response_source")
        response = diagnostics.get("response", "")
        
        # Build reflection summary
        reflection = {
            "query": query,
            "timestamp": datetime.utcnow().isoformat(),
            "reasoning_steps": len(reasoning_thoughts),
            "confidence_prior": confidence_prior,
            "fiber_harmony_score": fiber_harmony_score,
            "emergence_score": emergence_score,
            "reward_push": reward_push,
            "response_source": response_source,
            "response_preview": (response or "")[:200],
            "thought_signature": {
                "reasoning_depth": diagnostics.get("recursive_depth", 0),
                "max3d_synthesis": diagnostics.get("max3d_synthesis_response") is not None,
                "analogical": diagnostics.get("analogical_generalization") is not None,
                "critic_divergent": diagnostics.get("critic_divergent", False)
            }
        }
        
        # Log self study
        _save_log(reflection)
        
        # Update thought index for pattern retrieval
        qcan = str(query).lower().strip()
        if qcan not in THOUGHT_INDEX:
            THOUGHT_INDEX[qcan] = []
        THOUGHT_INDEX[qcan].append({
            "timestamp": reflection["timestamp"],
            "confidence_prior": confidence_prior,
            "fiber_harmony_score": fiber_harmony_score,
            "response_source": response_source
        })
        # Keep last 50 entries per query
        if len(THOUGHT_INDEX[qcan]) > 50:
            THOUGHT_INDEX[qcan] = THOUGHT_INDEX[qcan][-50:]
        
        # Persist index
        try:
            with open(THOUGHT_INDEX_PATH, "w", encoding="utf-8") as f:
                json.dump(THOUGHT_INDEX, f)
        except Exception:
            pass
            
        return reflection
    except Exception:
        return {"query": query, "error": "self_study_failed"}

def get_thought_history(query):
    qcan = str(query).lower().strip()
    return THOUGHT_INDEX.get(qcan, [])
