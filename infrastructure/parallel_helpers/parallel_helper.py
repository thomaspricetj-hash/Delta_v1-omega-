"""
Parallel helper for orchestrator stages.
Provides micro-helper utilities for concurrent execution.
"""
import concurrent.futures as cf
import time

def run_parallel(tasks, max_workers=4, timeout=30):
    """
    tasks: list of callables returning dict with response_source/response/confidence
    Returns best result by confidence.
    """
    best = None
    best_conf = -1
    with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(lambda fn: fn()): fn for fn in tasks}
        for fut in cf.as_completed(futures, timeout=timeout):
            try:
                res = fut.result()
                if res and res.get('response'):
                    conf = res.get('confidence',0)
                    if conf > best_conf:
                        best_conf = conf
                        best = res
            except Exception:
                pass
    return best

def micro_helper(stage_fn, query, diagnostics):
    """Micro helper wraps a stage with timing and error handling."""
    start = time.perf_counter()
    try:
        res = stage_fn(query, diagnostics)
        return {"result": res, "latency": time.perf_counter()-start, "error": None}
    except Exception as e:
        return {"result": None, "latency": time.perf_counter()-start, "error": str(e)}
