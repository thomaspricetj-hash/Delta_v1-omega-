"""
Background consumer for Memory Pool
Drains orchestrator results and teach events for consolidation
"""
import sys, time, json
from pathlib import Path
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "orchestrator"))
sys.path.insert(0, str(BASE / "infrastructure" / "memory_pool"))

from memory_pool import get_memory_pool

pool = get_memory_pool()
log_path = BASE / "runtime" / "logs" / "memory_pool_consumer.jsonl"
log_path.parent.mkdir(parents=True, exist_ok=True)

def process_item(item):
    t = item.get("type")
    try:
        if t == "orchestrator_result":
            query = item.get("query")
            diag = item.get("diag", {})
            # Example: promote high confidence hits
            src = diag.get("response_source")
            conf = diag.get("confidence_prior", 0.5)
            if src in ["context_cache_fast_gate", "fc_mbfi_fact_core"] and conf > 0.85:
                # Could trigger promotion, here we just log
                pass
        elif t == "teach_event":
            # Could trigger third-brain braiding, self-study, etc.
            pass
        # Write a tiny audit log
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": time.time(), "type": t}) + "\n")
    except Exception:
        pass

print("Memory Pool consumer started. Press Ctrl+C to stop.")
while True:
    item = pool.get(timeout=1.0)
    if item:
        process_item(item)
    # light sleep to avoid busy loop
    time.sleep(0.01)
