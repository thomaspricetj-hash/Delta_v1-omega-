"""
Supervisor for Translation OS v1
Oversees orchestrator, task manager, governor and checklist.
"""
import time, threading
from pathlib import Path
BASE = Path(__file__).resolve().parent.parent.parent
RUNTIME = BASE / "runtime" / "logs"
SUP_LOG = RUNTIME / "supervisor.jsonl"

class Supervisor:
    def __init__(self):
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        self.offload_thread = threading.Thread(target=self._offload_loop, daemon=True)
        self.offload_thread.start()

    def _loop(self):
        while self.running:
            try:
                # Health check
                from bitdrop_persistence import save_line
                # Load task manager stats
                from infrastructure.task_manager.task_manager import get_task_manager
                tm = get_task_manager()
                # Simple health snapshot
                save_line(SUP_LOG, {"ts": time.time(), "event": "heartbeat", "queue_len": len(tm.queue)})
            except Exception:
                pass
            time.sleep(10)

    def _offload_loop(self):
        while self.running:
            try:
                time.sleep(120)  # 2 minutes
                self._offload_caches()
            except Exception:
                pass

    def _offload_caches(self):
        try:
            import sys, json
            from pathlib import Path
            BASE = Path(__file__).resolve().parent.parent.parent
            sys.path.insert(0, str(BASE / "infrastructure"))
            from bitdrop_v2_integration import compress_data
            from bitdrop_persistence import save_line
            offload_dir = BASE / "runtime" / "logs" / "cache_offload"
            offload_dir.mkdir(parents=True, exist_ok=True)
            # Offload context cache example
            try:
                sys.path.insert(0, str(BASE / "infrastructure" / "context_cache"))
                from context_cache.context_cache import get_context_cache
                ctx = get_context_cache()
                data = json.dumps({"type":"context_cache","ts":time.time()}).encode()
                comp = compress_data(data)
                save_line(offload_dir / "context_cache_offload.jsonl", {"ts":time.time(),"compressed_ratio":comp.get("ratio")})
            except Exception:
                pass
        except Exception:
            pass

    def shutdown(self):
        self.running = False

_supervisor = Supervisor()

def get_supervisor():
    return _supervisor
