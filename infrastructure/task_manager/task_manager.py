"""
Task Manager for Translation OS v1
Monitors system load and throttles learning tasks to avoid huge blips.
"""
import psutil, time, threading
from pathlib import Path
BASE = Path(__file__).resolve().parent.parent.parent
RUNTIME = BASE / "runtime" / "logs"
TASK_LOG = RUNTIME / "task_manager.jsonl"

class TaskManager:
    def __init__(self, cpu_soft=50, cpu_hard=90, io_soft=80):
        self.cpu_soft = cpu_soft
        self.cpu_hard = cpu_hard
        self.io_soft = io_soft
        self.queue = []
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()
        # Worker pool for CPU/GPU split
        sys_path = str(Path(__file__).resolve().parent.parent / "worker_pool")
        import sys
        sys.path.insert(0, sys_path)
        from worker_pool import get_worker_pool
        self.workers = get_worker_pool()

    def _monitor(self):
        while self.running:
            cpu = psutil.cpu_percent(interval=0.5)
            # log snapshot
            try:
                from bitdrop_persistence import save_line
                save_line(TASK_LOG, {"ts": time.time(), "cpu": cpu, "queue_len": len(self.queue)})
            except:
                pass
            # throttle
            if cpu > self.cpu_hard:
                time.sleep(1)
            elif cpu > self.cpu_soft:
                time.sleep(0.2)
            else:
                time.sleep(0.1)

    def submit(self, fn, *args, **kwargs):
        with self.lock:
            self.queue.append((fn, args, kwargs))
        self._run_next()

    def _run_next(self):
        cpu = psutil.cpu_percent(interval=None)
        if cpu > self.cpu_soft:
            return
        with self.lock:
            if not self.queue:
                return
            fn, args, kwargs = self.queue.pop(0)
        try:
            # Route to CPU/GPU workers based on task name
            name = getattr(fn, "__name__", "")
            if "gpu" in name.lower() or "compress" in name.lower():
                self.workers.submit_gpu(fn, *args, **kwargs)
            else:
                self.workers.submit_cpu(fn, *args, **kwargs)
        except Exception:
            pass

    def shutdown(self):
        self.running = False

_task_manager = TaskManager()

def get_task_manager():
    return _task_manager
