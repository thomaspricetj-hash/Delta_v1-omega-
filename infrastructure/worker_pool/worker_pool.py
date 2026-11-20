"""
Worker Pool for Translation OS v1
CPU/GPU split workers for reasoning tasks.
"""
import concurrent.futures, os
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent.parent
CPU_WORKERS = max(1, os.cpu_count() // 2)
GPU_WORKERS = max(1, os.cpu_count() // 2)

class WorkerPool:
    def __init__(self):
        self.cpu_pool = concurrent.futures.ThreadPoolExecutor(max_workers=CPU_WORKERS)
        self.gpu_pool = concurrent.futures.ThreadPoolExecutor(max_workers=GPU_WORKERS)

    def submit_cpu(self, fn, *args, **kwargs):
        return self.cpu_pool.submit(fn, *args, **kwargs)

    def submit_gpu(self, fn, *args, **kwargs):
        return self.gpu_pool.submit(fn, *args, **kwargs)

    def shutdown(self):
        self.cpu_pool.shutdown(wait=True)
        self.gpu_pool.shutdown(wait=True)

_worker_pool = WorkerPool()

def get_worker_pool():
    return _worker_pool
