"""
Critic / Red-Team agent
Re-runs answer via different compositional path and flags divergence
"""
import sys
from pathlib import Path
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "orchestrator"))
from master_orchestrator_v0 import run_orchestrator

def critic_check(query, original_answer):
    # Run with different path: force graph explain if applicable
    # Simple heuristic: add "explain" prefix
    alt_query = f"Explain {query}"
    res = run_orchestrator(alt_query)
    alt_answer = res.get("response") or ""
    # Divergence if both non-empty and not overlapping
    overlap = original_answer.lower() in alt_answer.lower() or alt_answer.lower() in original_answer.lower()
    divergent = bool(original_answer and alt_answer and not overlap)
    return {
        "original": original_answer,
        "alt": alt_answer,
        "divergent": divergent,
        "source": res.get("response_source")
    }

if __name__ == "__main__":
    q = "What is the capital of France?"
    r = run_orchestrator(q)
    print(critic_check(q, r.get("response")))
