"""
Policy Engine V0
Defines how Translation OS chooses between actions
"""
import json, sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
POLICY_PATH = BASE_DIR / "infrastructure" / "policy" / "policy_v0.json"

def load_policies():
    with open(POLICY_PATH) as f:
        return json.load(f)

def get_priorities():
    data = load_policies()
    return data.get("priorities", [])

def _load_emergence_score():
    try:
        sys.path.insert(0, str(BASE_DIR / "scripts"))
        from bitdrop_persistence import load_json_bitdrop
        hist = load_json_bitdrop(BASE_DIR / "runtime" / "logs" / "emergence_score_history.jsonl")
        return hist or {}
    except Exception:
        return {}

def select_mode(emergence_score, avg_confidence):
    # Simple policy: exploration when emergence low, exploitation when high
    if emergence_score < 0.4:
        return "exploration"
    elif avg_confidence < 0.5:
        return "teaching"
    elif emergence_score > 0.8:
        return "consolidation"
    else:
        return "exploitation"

def get_governor_overrides(mode):
    overrides = {
        "exploration": {"dream_cycles": 30, "synthetic_qa_batch": 400, "wikidata_batch_limit": 80000},
        "teaching": {"dream_cycles": 20, "synthetic_qa_batch": 300, "wikidata_batch_limit": 50000},
        "consolidation": {"dream_cycles": 10, "synthetic_qa_batch": 100, "wikidata_batch_limit": 20000},
        "exploitation": {"dream_cycles": 15, "synthetic_qa_batch": 200, "wikidata_batch_limit": 50000}
    }
    return overrides.get(mode, overrides["exploitation"])

def apply_policy(query, context):
    q = query.lower()
    policies = load_policies()
    active = []
    if "math-boost" in q or "prevent" in q or "blocking" in q:
        active.append("resolve_blockers_before_new_work")
    if "what is" in q or "define" in q:
        active.append("cache_before_model")
        active.append("reuse_before_rebuild")
    if "next" in q or "build" in q:
        active.append("resolve_blockers_before_new_work")
    # Mode selection based on global state
    try:
        from scripts.bitdrop_persistence import load_json_bitdrop
        hist = load_json_bitdrop(BASE_DIR / "runtime" / "logs" / "emergence_score_history.jsonl")
        emergence_score = hist.get("score", 0.5) if isinstance(hist, dict) else 0.5
    except Exception:
        emergence_score = 0.5
    avg_conf = context.get("avg_confidence", 0.5)
    mode = select_mode(emergence_score, avg_conf)
    # Log policy decision
    try:
        sys.path.insert(0, str(BASE_DIR / "infrastructure" / "master_map_integrator"))
        from master_map_integrator import log_ledger_entry
        log_ledger_entry("policy_decision", {"query": query[:200], "mode": mode, "active_policies": active})
    except Exception:
        pass
    return {
        "query": query,
        "mode": mode,
        "active_policies": active,
        "priorities": get_priorities(),
        "emergence_score": emergence_score,
        "avg_confidence": avg_conf
    }

if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is preventing Math-Boost?"
    print(json.dumps(apply_policy(q, {}), indent=2))
