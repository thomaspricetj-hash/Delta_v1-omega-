"""
Paged KV cache for autoregressive generation.
Simple CPU implementation with LRU eviction and sharded persistence.
"""
import time
from collections import OrderedDict
from pathlib import Path
import json, numpy as np

CACHE_DIR = Path(__file__).resolve().parents[2] / "runtime" / "logs" / "kv_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

class PagedKVCache:
    def __init__(self, max_items=10000, page_size=256):
        self.max_items = max_items
        self.page_size = page_size
        self.cache = OrderedDict()
        self.access = {}
        
    def _key(self, seq_id, layer, pos):
        return f"{seq_id}:{layer}:{pos}"
    
    def write(self, seq_id, layer, k, v):
        key = self._key(seq_id, layer, len(self.cache))
        self.cache[key] = {"k": np.array(k, dtype=np.float32), "v": np.array(v, dtype=np.float32), "ts": time.time()}
        self.access[key] = time.time()
        if len(self.cache) > self.max_items:
            oldest = min(self.access, key=self.access.get)
            self.cache.pop(oldest, None)
            self.access.pop(oldest, None)
    
    def read(self, seq_id, layer, pos):
        # Simple read by seq_id prefix
        for k, v in self.cache.items():
            if k.startswith(f"{seq_id}:{layer}:"):
                return v["k"], v["v"]
        return None, None
    
    def evict_old(self, max_age=3600):
        now = time.time()
        to_del = [k for k,t in self.access.items() if now - t > max_age]
        for k in to_del:
            self.cache.pop(k, None)
            self.access.pop(k, None)

_cache = PagedKVCache()

def get_kv_cache():
    return _cache
