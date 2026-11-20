"""
Nightly consolidation of reasoning chains into weave lattice and convergence map.
"""
import json
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "orchestrator"))
from master_orchestrator_v0 import run_orchestrator

LOG_PATH = BASE / "runtime" / "logs" / "orchestrator_diagnostics.jsonl"

def consolidate_day():
    if not LOG_PATH.exists():
        return
    chains = []
    for line in LOG_PATH.read_text().splitlines():
        try:
            e = json.loads(line)
            diag = e.get("diagnostics", {})
            chain = diag.get("reasoning_chain", [])
            if chain:
                chains.append({"query": e.get("query_original"), "chain": chain})
        except:
            continue
    # Merge into master map and convergence map
    sys.path.insert(0, str(BASE / "runtime" / "master_map"))
    from master_map import get_master_map
    mm = get_master_map()
    sys.path.insert(0, str(BASE / "infrastructure" / "convergence_map"))
    from convergence_map import add_reward_signal
    sys.path.insert(0, str(BASE / "infrastructure" / "weave"))
    from lattice_weave_interference import weave_layers
    
    for entry in chains:
        q = entry["query"]
        for step in entry["chain"]:
            r = step.get("retrieved")
            if q and r:
                mm.update(q, r, problem_type="nightly_consolidation", steps=["chain"], success=True, source="nightly", weight=0.7)
                weave_layers(q, r, weight=0.7)
                add_reward_signal(q, r, reward=0.5)
    print(f"Consolidated {len(chains)} reasoning chains")

if __name__ == "__main__":
    consolidate_day()
