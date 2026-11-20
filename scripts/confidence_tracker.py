"""
Confidence & Self-Correction with Snapshot Memory
"""
import json
from pathlib import Path
BASE = Path(__file__).resolve().parent.parent
STATE_PATH = BASE / 'runtime' / 'logs' / 'confidence_state.json'

def load():
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {}

def save(s):
    STATE_PATH.write_text(json.dumps(s))

def update_confidence(query, answer, success):
    state = load()
    key = query.lower()
    if key not in state:
        state[key] = {"answer": answer, "confidence": 0.5, "successes": 0, "attempts": 0}
    state[key]["attempts"] += 1
    if success:
        state[key]["successes"] += 1
        state[key]["confidence"] = min(1.0, state[key]["confidence"] + 0.1)
    else:
        state[key]["confidence"] = max(0.0, state[key]["confidence"] - 0.1)
    state[key]["answer"] = answer
    save(state)
    return state[key]["confidence"]

def get_confidence(query):
    state = load()
    return state.get(query.lower(), {}).get("confidence", 0.5)
