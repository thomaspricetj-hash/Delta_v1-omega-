"""
Closed-loop self-correction with Snapshot Memory
"""
import json, sys
from pathlib import Path
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "orchestrator"))
from master_orchestrator_v0 import run_orchestrator, teach_full

CONF_PATH = BASE / "runtime" / "logs" / "confidence_state.json"
THRESHOLD = 0.5

def load_conf():
    try:
        return json.loads(CONF_PATH.read_text())
    except:
        return {}

def update_conf(query, answer, success):
    state = load_conf()
    key = query.lower()
    if key not in state:
        state[key] = {"answer": answer, "confidence": 0.5, "successes":0, "attempts":0}
    state[key]["attempts"] += 1
    if success:
        state[key]["successes"] += 1
        state[key]["confidence"] = min(1.0, state[key]["confidence"] + 0.1)
    else:
        state[key]["confidence"] = max(0.0, state[key]["confidence"] - 0.1)
    state[key]["answer"] = answer
    CONF_PATH.write_text(json.dumps(state))

def self_correct(query, answer):
    # Internal consistency check: re-run with reformulation
    reform = query + " explain in different words"
    res2 = run_orchestrator(reform)
    ans2 = res2.get("response") or ""
    # Simple divergence check
    success = answer.lower() in ans2.lower() or ans2.lower() in answer.lower() or not ans2
    if not success:
        # Try to teach corrected answer back
        teach_full(query, ans2)
        update_conf(query, ans2, True)
        return ans2, True
    return answer, False

if __name__ == "__main__":
    # Demo
    q = "What is the capital of France?"
    r = run_orchestrator(q)
    ans = r.get("response") or "Paris"
    corr_ans, changed = self_correct(q, ans)
    print("Original:", ans)
    print("Corrected:", corr_ans, "changed:", changed)
