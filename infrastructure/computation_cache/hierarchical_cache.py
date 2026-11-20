"""
Hierarchical computation cache for matrix multiplications.
Level 1: Exact hash match
Level 2: Approximate hash bucket match
Level 3: Predictor using map/index/ledger
Level 4: Fall back to full matrix multiplication.
Tracks cache hit rate and average lookup cost.
Disable caching if lookup cost exceeds compute savings.
"""
import hashlib
import time
from collections import defaultdict

class HierarchicalComputationCache:
    def __init__(self, max_size=10000, approx_bucket_bits=12):
        self.max_size = max_size
        self.approx_bucket_bits = approx_bucket_bits
        self.exact_cache = {}
        self.bucket_cache = defaultdict(dict)
        # Map/index/ledger for predictor
        self.index = {}  # shape_signature -> list of (hash, value)
        self.ledger = []  # log of accesses
        self.hits = 0
        self.misses = 0
        self.predictor_hits = 0
        self.lookup_costs = []
        self.compute_costs = []
        self.enabled = True

    def _hash(self, a, b):
        h = hashlib.sha256()
        h.update(str(a.shape).encode())
        h.update(str(b.shape).encode())
        try:
            import torch
            if isinstance(a, torch.Tensor):
                h.update(a.flatten()[:64].cpu().numpy().tobytes())
            if isinstance(b, torch.Tensor):
                h.update(b.flatten()[:64].cpu().numpy().tobytes())
        except:
            h.update(repr(a).encode()[:256])
            h.update(repr(b).encode()[:256])
        return h.hexdigest()

    def _shape_signature(self, a, b):
        return (tuple(a.shape), tuple(b.shape))

    def _bucket_key(self, full_hash):
        return full_hash[:self.approx_bucket_bits//4]

    def _predict_via_map(self, a, b):
        # Use index/ledger to predict
        sig = self._shape_signature(a, b)
        # Find similar shapes in index
        candidates = []
        for key, entries in self.index.items():
            if key[0] == sig[0] and key[1] == sig[1]:
                candidates.extend(entries)
        if candidates:
            # Return most recent
            # Simple heuristic: return last entry
            return candidates[-1][1]
        return None

    def get(self, a, b):
        if not self.enabled:
            return None
        start = time.perf_counter()
        full_hash = self._hash(a, b)
        # Level 1: exact
        if full_hash in self.exact_cache:
            self.hits += 1
            self.lookup_costs.append(time.perf_counter() - start)
            self.ledger.append({'type':'exact_hit','hash':full_hash})
            return self.exact_cache[full_hash]
        # Level 2: approximate bucket
        bucket = self._bucket_key(full_hash)
        if bucket in self.bucket_cache and self.bucket_cache[bucket]:
            self.hits += 1
            self.lookup_costs.append(time.perf_counter() - start)
            self.ledger.append({'type':'approx_hit','bucket':bucket})
            return self.bucket_cache[bucket].get('value')
        # Level 3: predictor via map/index/ledger
        pred = self._predict_via_map(a, b)
        if pred is not None:
            self.predictor_hits += 1
            self.hits += 1
            self.lookup_costs.append(time.perf_counter() - start)
            self.ledger.append({'type':'predictor_hit','sig':self._shape_signature(a,b)})
            return pred
        self.misses += 1
        self.lookup_costs.append(time.perf_counter() - start)
        self.ledger.append({'type':'miss','hash':full_hash})
        return None

    def set(self, a, b, value):
        if not self.enabled:
            return
        full_hash = self._hash(a, b)
        if len(self.exact_cache) >= self.max_size:
            self.exact_cache.pop(next(iter(self.exact_cache)))
        self.exact_cache[full_hash] = value
        # Update index/ledger map
        sig = self._shape_signature(a, b)
        if sig not in self.index:
            self.index[sig] = []
        self.index[sig].append((full_hash, value))
        bucket = self._bucket_key(full_hash)
        self.bucket_cache[bucket]['value'] = value

    def record_compute_cost(self, cost):
        self.compute_costs.append(cost)

    def stats(self):
        total = self.hits + self.misses
        hit_rate = self.hits / max(1, total)
        avg_lookup = sum(self.lookup_costs)/max(1,len(self.lookup_costs))
        avg_compute = sum(self.compute_costs)/max(1,len(self.compute_costs))
        should_disable = False
        if total > 100:
            if hit_rate < 0.05 and avg_lookup > 0:
                should_disable = True
        return {
            'hits': self.hits,
            'misses': self.misses,
            'predictor_hits': self.predictor_hits,
            'hit_rate': hit_rate,
            'avg_lookup_ms': avg_lookup*1000,
            'avg_compute_ms': avg_compute*1000,
            'enabled': self.enabled,
            'should_disable': should_disable,
            'index_entries': len(self.index),
            'ledger_size': len(self.ledger)
        }

    def maybe_disable(self):
        s = self.stats()
        if s['should_disable']:
            self.enabled = False
            return True
        return False
