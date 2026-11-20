"""
North Star – Stable reference for orientation and drift correction
Provides long-term alignment vector, goal anchor, and reward shaping.
"""
import json
from pathlib import Path
BASE = Path(__file__).resolve().parent.parent.parent
RUNTIME_LOGS = BASE / "runtime" / "logs"
STATE_PATH = RUNTIME_LOGS / "north_star_state.json"

def load_state():
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text())
        except:
            pass
    # Default stable references
    return {
        "anchor_vector": {"x":1.0,"y":0.5,"z":0.3},
        "core_goals": ["learn continuously","maintain coherence","generalize safely"],
        "user_preferences": {},
        "alignment_history": []
    }

def save_state(state):
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state))

def update_preferences(prefs):
    state = load_state()
    state["user_preferences"].update(prefs)
    save_state(state)

def get_alignment(query_pos, fiber_score=0.5, confidence=0.5):
    state = load_state()
    anchor = state["anchor_vector"]
    # cosine-like alignment
    dot = query_pos["x"]*anchor["x"] + query_pos["y"]*anchor["y"] + query_pos["z"]*anchor["z"]
    mag_q = (query_pos["x"]**2 + query_pos["y"]**2 + query_pos["z"]**2)**0.5 + 1e-9
    mag_a = (anchor["x"]**2 + anchor["y"]**2 + anchor["z"]**2)**0.5
    alignment = max(0.0, min(1.0, dot/(mag_q*mag_a)))
    # drift correction
    correction = {
        "dx": anchor["x"] - query_pos["x"],
        "dy": anchor["y"] - query_pos["y"],
        "dz": anchor["z"] - query_pos["z"]
    }
    # reward shaping
    reward = alignment * (0.5 + 0.5*fiber_score) * (0.5 + 0.5*confidence)
    # log
    state["alignment_history"].append({
        "alignment": alignment,
        "reward": reward,
        "ts": __import__('time').time()
    })
    state["alignment_history"] = state["alignment_history"][-1000:]
    save_state(state)
    return {
        "alignment": alignment,
        "correction": correction,
        "reward": reward,
        "anchor": anchor
    }

def drift_correction(pos):
    # Pull position toward anchor with small step
    state = load_state()
    a = state["anchor_vector"]
    alpha = 0.05
    return {
        "x": pos["x"] + alpha*(a["x"]-pos["x"]),
        "y": pos["y"] + alpha*(a["y"]-pos["y"]),
        "z": pos["z"] + alpha*(a["z"]-pos["z"])
    }
