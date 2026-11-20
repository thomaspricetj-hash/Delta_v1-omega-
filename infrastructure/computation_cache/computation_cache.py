"""
Computation Cache with Pattern Analyzer
Hash matrix blocks, store patterns, reuse multiplication outputs.
Tracks hit rate and evicts low-value entries.
"""
import hashlib
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Tuple

class ComputationCache:
    def __init__(self, capacity: int = 10000, persist_path: Path = None):
        self.capacity = capacity
        self.cache = OrderedDict()
        self.stats = {"hits":0, "misses":0, "evictions":0}
        self.persist_path = persist_path
        self._load()

    def _hash_block(self, block):
        h = hashlib.sha256()
        h.update(json.dumps(block, sort_keys=True).encode())
        return h.hexdigest()

    def get(self, key):
        if key in self.cache:
            self.stats["hits"] += 1
            # Move to end for LRU
            self.cache.move_to_end(key)
            return self.cache[key]
        self.stats["misses"] += 1
        return None

    def put(self, key, value, value_score: float = 1.0):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = {"value": value, "score": value_score, "hits": self.cache.get(key, {}).get("hits",0)+1}
        try:
            from common_index.common_index import index_output
            index_output(key)
        except Exception:
            pass
        if len(self.cache) > self.capacity:
            self._evict()
        self._save()

    def _evict(self):
        # Evict lowest value entries
        if not self.cache:
            return
        # Simple LRU + score
        items = list(self.cache.items())
        items.sort(key=lambda kv: (kv[1].get("score",0), kv[1].get("hits",0)))
        to_evict = len(self.cache) - self.capacity + 1
        for k,_ in items[:to_evict]:
            del self.cache[k]
            self.stats["evictions"] += 1

    def hash_matrix_block(self, matrix):
        return self._hash_block(matrix)

    def store_pattern(self, matrix_block):
        hid = self.hash_matrix_block(matrix_block)
        if hid not in self.cache:
            # Store placeholder for precomputed transformation
            self.put(hid, {"pattern_id": hid, "block": matrix_block})
        return hid

    def reuse_multiplication(self, a_block, b_block):
        # Pattern Analyzer -> Pattern ID -> Precomputed Transformation
        key = (self.hash_matrix_block(a_block), self.hash_matrix_block(b_block))
        cached = self.get(str(key))
        if cached:
            return True, cached["value"]
        # Miss -> compute and store
        # Placeholder compute
        result = {"computed": True}
        self.put(str(key), result, value_score=1.0)
        return False, result

    def hit_rate(self):
        total = self.stats["hits"] + self.stats["misses"]
        return self.stats["hits"]/total if total else 0.0

    def save(self):
        if not self.persist_path:
            return
        data = {
            "cache": {k: v for k,v in self.cache.items()},
            "stats": self.stats
        }
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.persist_path, 'w') as f:
            json.dump(data, f)

    def _load(self):
        if self.persist_path and self.persist_path.exists():
            with open(self.persist_path) as f:
                data = json.load(f)
            self.cache = OrderedDict(data.get("cache", {}))
            self.stats = data.get("stats", self.stats)

# Singleton
_cache = ComputationCache(capacity=10000, persist_path=Path("TRANSLATION_OS_V1/infrastructure/computation_cache/cache.json"))

def get_cache():
    return _cache

def hash_block(block):
    return _cache.hash_matrix_block(block)

def store_pattern(block):
    return _cache.store_pattern(block)

def reuse_mul(a,b):
    return _cache.reuse_multiplication(a,b)

def hit_rate():
    return _cache.hit_rate()
