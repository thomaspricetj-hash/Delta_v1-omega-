"""
Day Dream - background problem solving
Design works out problems in the background during daytime idle
"""
import sys, time, json, random
from pathlib import Path
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "orchestrator"))
from master_orchestrator_v0 import run_orchestrator, teach_full

DAY_DREAM_LOG = BASE / "runtime" / "logs" / "day_dream.log"
PROBLEMS_PATH = BASE / "runtime" / "logs" / "progress_tracker.json"

def load_pending_problems():
    pending = []
    # First drain day dream queue from orchestrator
    from bitdrop_persistence import load_lines_bitdrop
    queue_path = BASE / "runtime" / "logs" / "day_dream_queue.jsonl"
    queue_items = load_lines_bitdrop(queue_path)
    for entry in queue_items[-50:]:
        if entry and entry.get("query"):
            pending.append({"type": "queued", "text": entry.get("query", "")})
    # Cluster-based batching: load clusters and pick representatives
    try:
        from chunking_clustering import CLUSTER_PATH
        from bitdrop_persistence import load_json_bitdrop
        cluster_data = load_json_bitdrop(CLUSTER_PATH)
        if cluster_data:
            # Prioritize micro clusters first
            micro_clusters = cluster_data.get("micro_clusters", [])
            for cl in micro_clusters:
                if len(cl) > 1:
                    pending.append({"type": "micro_cluster", "text": cl[0]})
            # Then micro groups
            micro_groups = cluster_data.get("micro_groups", [])
            for cl in micro_groups:
                if len(cl) > 1:
                    pending.append({"type": "micro_group", "text": cl[0]})
            # Then regular clusters
            clusters = cluster_data.get("clusters", [])
            for cl in clusters:
                if len(cl) > 1:
                    pending.append({"type": "cluster", "text": cl[0]})
    except Exception:
        pass
    # Fallback to future goals
    if not pending:
        try:
            from workspace_memory.future_queries import load_future_book
            fb = load_future_book()
            if fb.get("current_goal"):
                pending.append({"type": "goal", "text": fb["current_goal"]})
            for bg in fb.get("blocked_goals", []):
                pending.append({"type": "blocked", "text": bg.get("goal", "")})
        except:
            pass
    if not pending:
        pending = [
            {"type": "synthetic", "text": "Explain why the sky is blue"},
            {"type": "synthetic", "text": "What is the capital of Finland?"},
            {"type": "synthetic", "text": "Summarize the cat and mouse story"},
        ]
    # Filter out pipe-contaminated entries
    pending = [p for p in pending if '|' not in p.get('text','')]
    return pending

def _run_orchestrator_safe(q, timeout=15):
    try:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(run_orchestrator, q)
            return fut.result(timeout=timeout)
    except Exception:
        return {"response": "Timeout", "response_source": "timeout", "reasoning_thoughts": []}

def day_dream_cycle():
    print("  DayDream: loading problems", flush=True)
    problems = load_pending_problems()
    print(f"  DayDream: {len(problems)} problems loaded", flush=True)
    # Uncertainty-driven prioritization
    try:
        from uncertainty_sampler import sample_uncertain
        uncertain = sample_uncertain(5)
        # Convert queue items to problem format
        for item in uncertain:
            if item.get("query"):
                problems.append({"type":"uncertain","text":item.get("query")})
    except Exception:
        pass
    # Pick a few problems, bias toward uncertain
    sample_size = min(3, len(problems))
    print(f"  DayDream: sampling {sample_size} problems", flush=True)
    # Prefer uncertain first
    uncertain_probs = [p for p in problems if p.get("type")=="uncertain"]
    random_probs = [p for p in problems if p.get("type")!="uncertain"]
    pick_pool = uncertain_probs + random_probs
    for i, prob in enumerate(random.sample(pick_pool, sample_size), 1):
        q = prob["text"]
        print(f"  DayDream: problem {i}/{sample_size} -> {q[:80]}...", flush=True)
        # Run orchestrator to think through problem
        res = _run_orchestrator_safe(q)
        answer = res.get("response") or "No answer yet"
        # Log day dream work
        entry = {
            "timestamp": time.time(),
            "problem_type": prob["type"],
            "problem": q,
            "answer": answer,
            "source": res.get("response_source"),
            "reasoning_thoughts": res.get("reasoning_thoughts", [])
        }
        from bitdrop_persistence import save_line
        save_line(DAY_DREAM_LOG, entry)
        # Teach the solution back for reinforcement
        try:
            teach_full(q, answer)
        except:
            pass
        time.sleep(0.2)

if __name__ == "__main__":
    print("Starting day dream background problem solving...")
    while True:
        day_dream_cycle()
        print("Day dream cycle complete, sleeping 120s")
        time.sleep(120)
