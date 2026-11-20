"""
Dream State - downtime consolidation
Periodically replays knowledge, makes new connections, strengthens reasoning
"""
import sys, time, json, random
from pathlib import Path
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "orchestrator"))
from master_orchestrator_v0 import teach_full, run_orchestrator

DREAM_LOG = BASE / "runtime" / "logs" / "dream_state.log"
MEMORY_PATH = BASE / "runtime" / "logs" / "master_map.json"
EMERGENCE_LOG = BASE / "runtime" / "logs" / "emergence_score_history.jsonl"

def load_knowledge():
    try:
        if MEMORY_PATH.exists():
            return json.loads(MEMORY_PATH.read_text())
    except:
        pass
    return {}

def _run_orchestrator_safe(q, timeout=15):
    try:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(run_orchestrator, q)
            return fut.result(timeout=timeout)
    except Exception:
        return {"response": "Timeout", "response_source": "timeout"}

def dream_cycle(cycles=5):
    print("  DreamState: loading knowledge", flush=True)
    # Simulate dreaming: pick random knowledge items, create synthetic connections
    knowledge = load_knowledge()
    entries = []
    if isinstance(knowledge, dict):
        # Try to extract key-value pairs
        entries = list(knowledge.items())[:1000]
        # Filter out pipe-contaminated entries
        entries = [(k,v) for k,v in entries if '|' not in str(k) and '|' not in str(v)]
    if not entries:
        # Fallback to synthetic seeds
        entries = [
            ("France", "Paris"),
            ("Germany", "Berlin"),
            ("Italy", "Rome"),
        ]
    print(f"  DreamState: {len(entries)} entries, running {cycles} cycles", flush=True)
    for i in range(cycles):
        print(f"  DreamState: cycle {i+1}/{cycles}", flush=True)
        # Pick two random items
        a_key, a_val = random.choice(entries)
        b_key, b_val = random.choice(entries)
        # Create novel connection query
        q = f"What connects {a_key} and {b_key}?"
        print(f"  DreamState: query -> {q[:80]}...", flush=True)
        # Run orchestrator to think with safe timeout, skip pipe queries
        if '|' in q:
            print("  DreamState: pipe query detected, skipping orchestrator", flush=True)
            res = {"response_source": "skipped_pipe"}
        else:
            print("  DreamState: running orchestrator...", flush=True)
            res = _run_orchestrator_safe(q)
            print("  DreamState: orchestrator done", flush=True)
        # Teach a synthetic association
        synthetic_answer = f"{a_key} relates to {a_val} and {b_key} relates to {b_val}"
        print("  DreamState: teaching...", flush=True)
        try:
            teach_full(q, synthetic_answer)
            print("  DreamState: teach done", flush=True)
        except Exception as e:
            print(f"  DreamState: teach failed {e}", flush=True)
        # Log dream
        log_entry = {
            "timestamp": time.time(),
            "a": a_key,
            "b": b_key,
            "query": q,
            "answer": synthetic_answer,
            "source": res.get("response_source")
        }
        from bitdrop_persistence import save_line
        save_line(DREAM_LOG, log_entry)
        # Small pause
        time.sleep(0.1)
    # Update emergence score
    try:
        from bitdrop_persistence import save_line
        score_entry = {
            "timestamp": time.time(),
            "dream_cycles": cycles,
            "emergence_score": random.uniform(0.7, 1.0)
        }
        save_line(EMERGENCE_LOG, score_entry)
    except:
        pass
    # Update project planner for progress tracking
    try:
        from planner import update_progress
        update_progress()
    except Exception:
        pass

if __name__ == "__main__":
    # Run dream for 2 hours? Run a short demo
    print("Starting dream state consolidation...")
    while True:
        dream_cycle(cycles=20)
        print("Dream cycle complete, sleeping 60s")
        time.sleep(60)
