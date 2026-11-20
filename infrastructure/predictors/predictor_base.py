"""
Base predictor utilities: Bloom filter, LRU cache, reward logging, helpers.
All predictors should use these for consistency.
"""
import hashlib
import json
from pathlib import Path
from time import perf_counter

BASE = Path(__file__).resolve().parents[2]
LOG_DIR = BASE / "runtime" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

class BloomFilter:
    def __init__(self, size=1<<22, hashes=4):
        self.size = size
        self.hashes = hashes
        self.bits = bytearray((size + 7)//8)
    def _hashes(self, item):
        data = item.encode('utf-8')
        h1 = int(hashlib.sha256(data).hexdigest(), 16)
        h2 = int(hashlib.sha256(data[::-1]).hexdigest(), 16)
        for i in range(self.hashes):
            yield (h1 + i*h2) % self.size
    def add(self, item):
        for pos in self._hashes(item):
            self.bits[pos//8] |= 1 << (pos % 8)
    def __contains__(self, item):
        return all((self.bits[pos//8] >> (pos % 8)) & 1 for pos in self._hashes(item))

class PredictorCache:
    def __init__(self, name, max_ram=5000, lru_max=256):
        self.name = name
        self.cache = {}
        self.bloom = BloomFilter()
        self.lru = {}
        self.lru_order = []
        self.max_ram = max_ram
        self.lru_max = lru_max
        self.hits = 0
        self.misses = 0
    def get(self, key):
        if key in self.bloom and key in self.cache:
            self.hits += 1
            self._touch_lru(key)
            return self.cache[key], True
        self.misses += 1
        return None, False
    def set(self, key, value):
        self.bloom.add(key)
        self.cache[key] = value
        self._touch_lru(key)
        if len(self.cache) > self.max_ram:
            # evict oldest
            old_key = next(iter(self.cache))
            self.cache.pop(old_key, None)
    def _touch_lru(self, key):
        if key in self.lru:
            self.lru_order.remove(key)
        self.lru_order.append(key)
        if len(self.lru_order) > self.lru_max:
            old = self.lru_order.pop(0)
            self.lru.pop(old, None)
    def stats(self):
        total = self.hits + self.misses
        hit_rate = self.hits / total if total else 0.0
        return {"hits": self.hits, "misses": self.misses, "hit_rate": hit_rate}

def log_predictor_event(predictor_name, event, data):
    try:
        path = LOG_DIR / "predictor_events.jsonl"
        entry = {"ts": perf_counter(), "predictor": predictor_name, "event": event, "data": data}
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass

def log_reward(predictor_name, reward, confidence, query):
    try:
        from infrastructure.predictors.lqr_reward_controller import update_reward
        update_reward(predictor_name, reward, confidence)
    except Exception:
        pass
    log_predictor_event(predictor_name, "reward", {"reward": reward, "confidence": confidence, "query": query[:200]})

def get_helpers():
    # Common helper utilities for predictors
    def decompose_problem(problem_str, threshold=1000):
        # Simple chunking by length
        if len(problem_str) <= threshold:
            return [problem_str]
        chunks = [problem_str[i:i+threshold] for i in range(0, len(problem_str), threshold)]
        return chunks
    def cluster_key(op, *args):
        return f"{op}:{':'.join(map(str, args))}"
    def estimate_confidence(hit, base=0.85):
        return min(0.99, base + 0.1 if hit else base - 0.2)
    return {
        "decompose_problem": decompose_problem,
        "cluster_key": cluster_key,
        "estimate_confidence": estimate_confidence,
    }
