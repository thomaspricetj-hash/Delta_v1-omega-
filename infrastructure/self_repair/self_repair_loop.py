"""
Self-repair loop for Translation OS v1
Monitors echo location low scores and triggers re-teaching
"""
import json, time, sys
from pathlib import Path

BASE = Path("TRANSLATION_OS_V1")
LOG_DIR = BASE / "runtime" / "logs"
ECHO_LOG = LOG_DIR / "echo_location.jsonl"

def _get_teach_full():
    import sys
    sys.path.insert(0, str(BASE / "orchestrator"))
    from master_orchestrator_v0 import teach_full
    return teach_full

def scan_and_repair():
    if not ECHO_LOG.exists():
        return
    seen = {}
    # read last 1000 entries
    entries = []
    with open(ECHO_LOG, "r", encoding="utf-8") as f:
        for line in f:
            try:
                e = json.loads(line)
                entries.append(e)
            except:
                pass
    entries = entries[-1000:]
    for e in entries:
        q = e.get("qcanon")
        score = e.get("location_score", 0)
        if score < 0.4:
            # trigger re-teach if not recently done
            last = seen.get(q, 0)
            if time.time() - last > 3600:
                # try to get answer from context cache
                try:
                    sys.path.insert(0, str(BASE / "infrastructure" / "context_cache"))
                    from context_cache import get_context_cache
                    cache = get_context_cache()
                    ans = cache.get(q)
                    if ans:
                        teach_full_fn = _get_teach_full()
                        teach_full_fn(e.get("query",""), ans, reward=0.5)
                        seen[q] = time.time()
                        print(f"[SelfRepair] Re-taught {q}")
                except Exception:
                    pass

if __name__ == "__main__":
    while True:
        scan_and_repair()
        time.sleep(300)
