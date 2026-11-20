"""
Memory Pool for parallel orchestrator stages.
Provides shared memory buffer between producers and consumers.
"""
import threading
from collections import deque
import time

class MemoryPool:
    def __init__(self, max_size=1000):
        self.max_size = max_size
        self.pool = deque(maxlen=max_size)
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)
    
    def put(self, item):
        with self.cond:
            self.pool.append({"item": item, "ts": time.time()})
            self.cond.notify()
    
    def get(self, timeout=1.0):
        with self.cond:
            if not self.pool:
                self.cond.wait(timeout)
            if self.pool:
                return self.pool.popleft()["item"]
            return None
    
    def size(self):
        with self.lock:
            return len(self.pool)

_pool = MemoryPool()

def get_memory_pool():
    return _pool
