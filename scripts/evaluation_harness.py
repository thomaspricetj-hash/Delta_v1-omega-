"""
Evaluation harness - automate 10 emergence tests daily
"""
import sys
from pathlib import Path
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "orchestrator"))
from master_orchestrator_v0 import run_orchestrator
import json, datetime

EVAL_LOG = BASE / "runtime" / "logs" / "emergence_eval.jsonl"

TESTS = {
    "Novel Combination": "What would happen if a submarine lived in a forest and learned astronomy?",
    "Consistency": "My favorite color is purple.",
    "Compression": "Summarize in one sentence: The cat chased the mouse. The mouse escaped.",
    "Abstraction": "What do dogs and fish have in common?",
    "Generalization": "Red apples are sweet. Would a red pear likely be sweet?",
    "Unseen": "A Flarn is larger than a Wex. A Wex is larger than a Nib. Which is largest?",
}

def _safe_run(q, timeout=20):
    try:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(run_orchestrator, q)
            return fut.result(timeout=timeout)
    except Exception:
        return {"response": "Timeout", "response_source": "timeout"}

def run_eval():
    results = []
    for name, q in TESTS.items():
        d = _safe_run(q)
        results.append({
            "test": name,
            "query": q,
            "response": d.get("response"),
            "source": d.get("response_source"),
            "has_response": d.get("response") is not None
        })
    from bitdrop_persistence import save_line
    entry = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "results": results
    }
    save_line(EVAL_LOG, entry)
    return entry

if __name__ == "__main__":
    print(json.dumps(run_eval(), indent=2))
