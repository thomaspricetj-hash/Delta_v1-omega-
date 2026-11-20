"""
Context Cache with decay and size cap
Most used items stay, decay reduces score over time
Never over-fills via max items + decay eviction
"""
import time
import threading
from collections import OrderedDict

import json, os
from pathlib import Path

CACHE_FILE = Path(__file__).parent / "context_cache_store.json"

class ContextCache:
    def __init__(self, max_items=10000, decay_half_life_seconds=3600):
        self.max_items = max_items
        self.half_life = decay_half_life_seconds
        self.layers = [OrderedDict()]
        self.layer_sizes = [max_items]
        self._lock = threading.Lock()
        self._access_counter = 0
        self._store = OrderedDict()
        self._load()
        self._compress_enabled = False

    def _load_from_offload(self, key):
        try:
            from pathlib import Path
            base = Path(__file__).resolve().parent.parent.parent
            offload_dir = base / "runtime" / "logs" / "cache_offload"
            if not offload_dir.exists():
                return None
            # Simple scan for matching key in offload logs
            for f in offload_dir.glob("*.jsonl"):
                try:
                    for line in f.open():
                        entry = json.loads(line)
                        if entry.get("key") == key:
                            # Rehydrate entry
                            return {
                                'value': entry.get('value'),
                                'score': entry.get('score',1.0),
                                'last_access': time.time(),
                                'compressed': entry.get('compressed', False)
                            }
                except Exception:
                    continue
        except Exception:
            pass
        return None

    def _load(self):
        try:
            if CACHE_FILE.exists():
                data = json.loads(CACHE_FILE.read_text())
                for k,v in data.items():
                    self._store[k] = v
        except Exception:
            pass

    def _save(self):
        try:
            # Only save serializable parts
            serial = {}
            for k,v in self._store.items():
                try:
                    serial[k] = {
                        'value': v['value'] if isinstance(v['value'], str) else v['value'].hex(),
                        'score': v['score'],
                        'last_access': v['last_access'],
                        'compressed': v.get('compressed', False)
                    }
                except Exception:
                    continue
            CACHE_FILE.write_text(json.dumps(serial))
        except Exception:
            pass
        try:
            import bitdrop_v2 as _bd2
            self._bd2 = _bd2
            self._compress_enabled = True
        except Exception:
            self._bd2 = None

    def _decay_score(self, entry):
        age = time.time() - entry['last_access']
        # exponential decay
        decay_factor = 0.5 ** (age / self.half_life)
        return entry['score'] * decay_factor

    def get(self, key):
        with self._lock:
            if key not in self._store:
                # Smart load-on-miss from offload storage
                loaded = self._load_from_offload(key)
                if loaded is not None:
                    self._store[key] = loaded
                    self._store.move_to_end(key)
                else:
                    return None
            entry = self._store[key]
            # update access stats
            self._access_counter += 1
            entry['score'] += 1.0  # usage boost
            entry['last_access'] = time.time()
            # move to end for LRU
            self._store.move_to_end(key)
            val = entry['value']
            # Decompress if stored compressed
            if entry.get('compressed') and self._compress_enabled:
                try:
                    # Try decompress via bitdrop_v2 if available
                    decompress_fn = getattr(self._bd2, 'decompress_adaptive_py', None)
                    if decompress_fn and isinstance(val, bytes):
                        val = decompress_fn(val).decode('utf-8', errors='ignore')
                except Exception:
                    pass
            return val

    def put(self, key, value, initial_score=1.0):
        with self._lock:
            now = time.time()
            # Try compress with BitDrop v2 if available
            compressed = False
            stored_value = value
            if self._compress_enabled and isinstance(value, str):
                try:
                    data = value.encode('utf-8')
                    comp = self._bd2.compress_adaptive_py(data)
                    stored_value = comp  # store compressed bytes
                    compressed = True
                except Exception:
                    compressed = False
            if key in self._store:
                entry = self._store[key]
                entry['score'] += initial_score
                entry['value'] = stored_value
                entry['last_access'] = now
                entry['compressed'] = compressed
                self._store.move_to_end(key)
            else:
                self._store[key] = {
                    'value': stored_value,
                    'score': initial_score,
                    'last_access': now,
                    'compressed': compressed
                }
            self._evict_if_needed()
            self._save()

    def _evict_if_needed(self):
        # Decay scores first
        now = time.time()
        # prune by decay and LRU
        items_to_remove = []
        for k, entry in list(self._store.items()):
            decayed = self._decay_score(entry)
            if decayed < 0.01:
                items_to_remove.append(k)
        for k in items_to_remove:
            self._store.pop(k, None)
        # enforce max items by score
        while len(self._store) > self.max_items:
            # find lowest decayed score
            worst_key = min(self._store.keys(), key=lambda k: self._decay_score(self._store[k]))
            self._store.pop(worst_key, None)

    def stats(self):
        with self._lock:
            return {
                'size': len(self._store),
                'max_items': self.max_items,
                'half_life': self.half_life
            }

# Singleton
_context_cache = ContextCache(max_items=1000, decay_half_life_seconds=7200)

def get_context_cache():
    return _context_cache
