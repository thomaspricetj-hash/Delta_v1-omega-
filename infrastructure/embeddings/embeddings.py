"""
Native embeddings for Translation OS v1
Simple incremental embedding store with cosine similarity attention.
"""
# print("[EMBEDDINGS MODULE] import start")
from __future__ import annotations
import os
import json
import hashlib
import sys
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional

BASE = Path(__file__).resolve().parents[2]
STORE_DIR = BASE / "runtime" / "embeddings"
STORE_DIR.mkdir(parents=True, exist_ok=True)
STORE_JSON = STORE_DIR / "embed_store.jsonl"
STORE_BD2 = STORE_DIR / "embed_store.bd2"
STORE_META = STORE_DIR / "embed_store_meta.json"
WEAVE_INDEX = STORE_DIR / "weave_index.json"
LEDGER_PATH = STORE_DIR / "embed_ledger.jsonl"
SHARD_DIR = STORE_DIR / "shards"
INDEX_PATH = STORE_DIR / "embed_store_index.json"
MAP_PATH = STORE_DIR / "embed_store_map.json"
LEDGER_IDX_PATH = STORE_DIR / "embed_store_ledger.jsonl"
GRAPH_PATH = STORE_DIR / "embed_store_graph.json"
HEATMAP_PATH = STORE_DIR / "embed_heatmap.json"

DIM = 256
SEED = 42

def _stable_seed(text: str) -> int:
    h = hashlib.sha256(text.encode()).digest()
    return int.from_bytes(h[:4], 'little') % (2**32)

def _init_vector(text: str) -> np.ndarray:
    rng = np.random.default_rng(_stable_seed(text) ^ SEED)
    v = rng.normal(0, 1, DIM).astype(np.float32)
    norm = np.linalg.norm(v) + 1e-9
    return v / norm

class EmbeddingStore:
    def __init__(self):
        # print("[EMBEDDINGS] __init__ start")
        self._vectors = {}
        self._access_order = []
        self._max_ram_items = 5000  # initial
        self._ram_guard_target = 0.35  # target RAM usage fraction
        self._weave_index = {}  # key -> {layer, neighbors, access_count}
        self._weave_layers = {}  # layer_id -> set(keys)
        self._shard_cache = {}  # shard_id -> {key: vec}
        self._index_map = {}  # key -> shard_id
        self._ledger_map = {}  # key -> {shard_id, line_num}
        self._shard_meta = {}
        # Multilayered heatmap with cold overlay
        self._heat = {}  # key -> heat score
        self._heat_decay = 0.995
        self._heat_hot_threshold = 0.5
        self._heat_cold_threshold = 0.01
        self._heatmap_layers = {}  # layer_id -> {key: heat}
        # Skim layer for cold embeddings
        self._skim_threshold = 0.05
        self._skim_hits = 0
        self._skim_skips = 0
        # Bloom filter for fast negative membership
        self._bloom = self._init_bloom(size=1<<20, hashes=3)
        # print("[EMBEDDINGS] loading weave index")
        self._load_weave_index()
        # print("[EMBEDDINGS] loading shard index")
        self._load_shard_index()
        # print("[EMBEDDINGS] loading store")
        self._load()
        self._warm_up_done = False
        # print("[EMBEDDINGS] init done")
        # Start background RAM guard adjuster
        self._start_ram_guard()
        # Load persisted heatmap if exists
        self._load_heatmap()
        # Start heatmap decay & persistence
        self._start_heatmap_daemon()
        # Warm-up heatmap from existing shards for immediate coverage
        self._warmup_heat_from_shards()

    def _start_ram_guard(self):
        try:
            import threading, time, psutil
            def _adjust():
                while True:
                    try:
                        mem = psutil.virtual_memory()
                        usage = mem.percent / 100.0
                        # Adjust max items based on RAM pressure
                        if usage > self._ram_guard_target + 0.05:
                            self._max_ram_items = max(500, int(self._max_ram_items * 0.9))
                        elif usage < self._ram_guard_target - 0.05:
                            self._max_ram_items = int(self._max_ram_items * 1.1)
                        # Enforce current limit
                        while len(self._vectors) > self._max_ram_items:
                            oldest = self._access_order.pop(0) if self._access_order else None
                            if oldest and oldest in self._vectors:
                                del self._vectors[oldest]
                    except Exception:
                        pass
                    time.sleep(30)
            t = threading.Thread(target=_adjust, daemon=True)
            t.start()
        except Exception:
            pass

    def _init_bloom(self, size=1<<20, hashes=3):
        # Simple Bloom filter using bytearray
        class Bloom:
            def __init__(self, size, hashes):
                self.size = size
                self.hashes = hashes
                self.bits = bytearray((size + 7) // 8)
            def _indices(self, key):
                import hashlib
                h = hashlib.sha256(key.encode()).digest()
                # Use first 8 bytes as base
                base = int.from_bytes(h[:8], 'little')
                indices = []
                for i in range(self.hashes):
                    # Different salt per hash
                    h2 = hashlib.sha256((key + str(i)).encode()).digest()
                    idx = (int.from_bytes(h2[:8], 'little') + base * i) % self.size
                    indices.append(idx)
                return indices
            def add(self, key):
                for idx in self._indices(key):
                    self.bits[idx // 8] |= 1 << (idx % 8)
            def __contains__(self, key):
                for idx in self._indices(key):
                    if not (self.bits[idx // 8] & (1 << (idx % 8))):
                        return False
                return True
        return Bloom(size, hashes)

    def _load_heatmap(self):
        try:
            bd2_path = HEATMAP_PATH.with_suffix('.json.bd2')
            data = None
            if bd2_path.exists():
                try:
                    from infrastructure.logging.bitdrop_log_writer import BitDropLogWriter
                    writer = BitDropLogWriter(HEATMAP_PATH.parent)
                    # Read first frame
                    with open(bd2_path, 'rb') as f:
                        import struct
                        hdr = f.read(4)
                        if len(hdr) == 4:
                            comp_len = struct.unpack('<I', hdr)[0]
                            comp = f.read(comp_len)
                            raw = writer._decompress(comp)
                            data = json.loads(raw.decode('utf-8'))
                except Exception:
                    data = None
            if data is None and HEATMAP_PATH.exists():
                data = json.loads(HEATMAP_PATH.read_text(encoding='utf-8'))
            if data:
                self._heat = data.get('heat', {})
                self._heatmap_layers = data.get('layers', {})
            else:
                self._heat = {}
                self._heatmap_layers = {}
        except Exception:
            self._heat = {}
            self._heatmap_layers = {}

    def _save_heatmap(self):
        try:
            data = {
                'heat': self._heat,
                'layers': self._heatmap_layers,
                'timestamp': __import__('time').time()
            }
            json_bytes = json.dumps(data).encode('utf-8')
            # Try BitDrop compression for heatmap persistence
            try:
                from infrastructure.logging.bitdrop_log_writer import BitDropLogWriter
                writer = BitDropLogWriter(HEATMAP_PATH.parent)
                bd2_path = HEATMAP_PATH.with_suffix('.json.bd2')
                # Write single compressed frame
                with open(bd2_path, 'wb') as f:
                    import struct
                    comp = writer._compress(json_bytes)
                    f.write(struct.pack('<I', len(comp)))
                    f.write(comp)
                # Keep plain json for fallback read
                # Optionally remove plain file to save space
                # if HEATMAP_PATH.exists(): HEATMAP_PATH.unlink()
            except Exception:
                # Fallback to plain JSON
                HEATMAP_PATH.write_text(json.dumps(data))
        except Exception:
            pass

    def _warmup_heat_from_shards(self):
        # Seed heatmap for existing shard keys so heatmap is populated on startup
        try:
            for sid, meta in self._shard_meta.items():
                keys = meta.get('keys', [])
                # Seed with small baseline heat to avoid cold start
                for k in keys:
                    if k not in self._heat:
                        self._heat[k] = 0.01
                        # Also populate bloom for fast negative check
                        try:
                            self._bloom.add(k)
                        except Exception:
                            pass
        except Exception:
            pass

    def _start_heatmap_daemon(self):
        try:
            import threading, time
            def _daemon():
                while True:
                    try:
                        self._decay_heat()
                        self._persist_meta_and_graph()
                    except Exception:
                        pass
                    time.sleep(60)  # update every minute
            t = threading.Thread(target=_daemon, daemon=True)
            t.start()
        except Exception:
            pass

    def _update_heat(self, key: str, delta: float = 0.1):
        # Increase heat on access, cap at 1.0
        h = self._heat.get(key, 0.0)
        h = min(1.0, h + delta)
        self._heat[key] = h
        # Update multilayered heatmap layers
        layer = self._weave_index.get(key, {}).get('layer', 0)
        if layer not in self._heatmap_layers:
            self._heatmap_layers[layer] = {}
        self._heatmap_layers[layer][key] = h

    def _decay_heat(self):
        # Exponential decay with cold overlay
        decayed = {}
        for k, h in self._heat.items():
            h_new = h * self._heat_decay
            if h_new < self._heat_cold_threshold:
                # Cold overlay: demote
                # Optionally remove from RAM if cold
                continue
            decayed[k] = h_new
        self._heat = decayed
        # Decay per-layer
        for layer_id, layer_dict in self._heatmap_layers.items():
            new_layer = {}
            for k, h in layer_dict.items():
                h_new = h * self._heat_decay
                if h_new >= self._heat_cold_threshold:
                    new_layer[k] = h_new
            self._heatmap_layers[layer_id] = new_layer

    def _chunk_cluster_shards(self, chunk_size=500, n_clusters=32):
        """Chunk, group and cluster shard keys for faster retrieval.
        - Chunking: split keys into fixed-size chunks for I/O batching
        - Grouping: group by weave layer / heat tier
        - Clustering: k-means on vector means to create hot clusters
        """
        try:
            import numpy as np
            from collections import defaultdict
            keys = list(self._heat.keys())
            if not keys:
                return
            # Chunking
            chunks = [keys[i:i+chunk_size] for i in range(0, len(keys), chunk_size)]
            # Grouping by layer and heat tier
            groups = defaultdict(list)
            for k in keys:
                layer = self._weave_index.get(k, {}).get('layer', 0)
                h = self._heat.get(k, 0.0)
                tier = 'hot' if h > self._heat_hot_threshold else 'warm' if h > self._heat_cold_threshold else 'cold'
                groups[(layer, tier)].append(k)
            # Simple clustering on available vectors
            # Use mean of first 8 dims as proxy for speed
            vecs = []
            valid_keys = []
            for k in keys:
                if k in self._vectors:
                    v = self._vectors[k]
                    vecs.append(v[:8])
                    valid_keys.append(k)
            if len(valid_keys) >= n_clusters:
                X = np.array(vecs, dtype=np.float32)
                # Mini k-means via numpy
                # Initialize centroids randomly
                rng = np.random.default_rng(42)
                centroids = X[rng.choice(len(X), size=n_clusters, replace=False)]
                for _ in range(5):
                    dists = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
                    labels = np.argmin(dists, axis=1)
                    for i in range(n_clusters):
                        mask = labels == i
                        if np.any(mask):
                            centroids[i] = X[mask].mean(axis=0)
                # Store cluster map for fast lookup
                self._cluster_map = {k: int(lbl) for k, lbl in zip(valid_keys, labels)}
            else:
                self._cluster_map = {}
            # Persist chunk/group metadata
            meta = {
                'chunks': len(chunks),
                'groups': {str(k): len(v) for k, v in groups.items()},
                'clusters': len(set(self._cluster_map.values())) if self._cluster_map else 0
            }
            # Save to meta file for inspection
            try:
                from pathlib import Path
                meta_path = STORE_DIR / "embed_chunk_meta.json"
                json.dump(meta, open(meta_path, 'w'))
            except Exception:
                pass
        except Exception:
            pass

    def _persist_meta_and_graph(self):
        try:
            # Run chunking / grouping / clustering for shard optimization
            self._chunk_cluster_shards(chunk_size=500, n_clusters=32)
            # Auto-tune skim threshold with PID-like heuristic
            total_keys = len(self._heat)
            if total_keys > 0:
                avg_heat = float(sum(self._heat.values())) / total_keys
                # Adjust skim threshold to keep ~20% of keys skimmed
                # Simple proportional control
                target_avg = 0.2
                error = target_avg - avg_heat
                # Move threshold inversely to avg heat
                self._skim_threshold = max(0.01, min(0.2, self._skim_threshold + error * 0.01))
            else:
                avg_heat = 0.0
            hot_keys = sum(1 for h in self._heat.values() if h >= self._heat_hot_threshold)
            cold_keys = sum(1 for h in self._heat.values() if h < self._heat_cold_threshold)
            # Estimate compression ratio from BitDrop store size
            try:
                store_size = STORE_BD2.stat().st_size if STORE_BD2.exists() else 0
                # Rough estimate: raw would be count * dim * 4 bytes
                raw_est = total_keys * DIM * 4
                compress_ratio = store_size / raw_est if raw_est else 0.0
            except Exception:
                compress_ratio = 0.0
            meta = {
                "dim": DIM,
                "quant": "int16",
                "count": total_keys,
                "hot_keys": hot_keys,
                "cold_keys": cold_keys,
                "avg_heat": avg_heat,
                "compress_ratio": compress_ratio,
                "layers": {str(k): len(v) for k, v in self._heatmap_layers.items()},
                "skim_hits": self._skim_hits,
                "skim_skips": self._skim_skips,
                "timestamp": __import__('time').time()
            }
            STORE_META.write_text(json.dumps(meta))
            # Persist embed_store_graph.json with heat-aware nodes
            nodes = []
            for sid, meta_shard in self._shard_meta.items():
                # Compute shard heat as average of keys in shard
                shard_keys = meta_shard.get('keys', [])
                shard_heat = 0.0
                if shard_keys:
                    heats = [self._heat.get(k, 0.0) for k in shard_keys]
                    shard_heat = sum(heats) / len(heats)
                nodes.append({
                    "id": int(sid),
                    "entries": meta_shard.get('entries', 0),
                    "heat": shard_heat
                })
            graph = {
                "nodes": nodes,
                "edges": [],  # Could be extended with co-access edges
                "layers": {str(k): len(v) for k, v in self._heatmap_layers.items()},
                "timestamp": __import__('time').time()
            }
            GRAPH_PATH.write_text(json.dumps(graph))
            # Persist heatmap for restart recovery
            self._save_heatmap()
        except Exception:
            pass

    def _load_weave_index(self):
        try:
            if WEAVE_INDEX.exists():
                data = json.loads(WEAVE_INDEX.read_text(encoding='utf-8'))
                self._weave_index = data.get('index', {})
                self._weave_layers = {int(k): set(v) for k,v in data.get('layers', {}).items()}
            else:
                self._weave_index = {}
                self._weave_layers = {}
        except Exception:
            self._weave_index = {}
            self._weave_layers = {}

    def _load_shard_index(self):
        try:
            if INDEX_PATH.exists():
                self._index_map = json.loads(INDEX_PATH.read_text(encoding='utf-8'))
            else:
                self._index_map = {}
            if MAP_PATH.exists():
                map_data = json.loads(MAP_PATH.read_text(encoding='utf-8'))
                self._shard_meta = map_data.get('shards', {})
            else:
                self._shard_meta = {}
            if LEDGER_IDX_PATH.exists():
                self._ledger_map = {}
                with open(LEDGER_IDX_PATH, 'r', encoding='utf-8') as f:
                    for line in f:
                        if not line.strip():
                            continue
                        entry = json.loads(line)
                        key = entry.get('key')
                        if key:
                            self._ledger_map[key] = {'shard_id': entry.get('shard_id'), 'line_num': entry.get('line_num')}
        except Exception:
            self._index_map = {}
            self._shard_meta = {}
            self._ledger_map = {}

    def _load_from_shard(self, shard_id, key):
        try:
            if shard_id not in self._shard_cache:
                meta = self._shard_meta.get(str(shard_id))
                if not meta:
                    return None
                shard_file = SHARD_DIR / f'embed_store_shard_{shard_id}.bd2.bd2'
                if not shard_file.exists():
                    return None
                # Decompress with BitDrop v2 if compressed
                try:
                    import bitdrop_v2 as bd
                    raw = shard_file.read_bytes()
                    # Try decompress, fallback to raw if already decompressed
                    try:
                        decompressed = bd.decompress_adaptive_py(raw)
                        text = decompressed.decode('utf-8')
                    except Exception:
                        text = raw.decode('utf-8')
                except Exception:
                    text = shard_file.read_text(encoding='utf-8')
                cache = {}
                for line in text.splitlines():
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    data = obj.get('data', {})
                    k = data.get('key')
                    vec_list = data.get('vec')
                    if k is None or vec_list is None:
                        continue
                    vec = np.array(vec_list, dtype=np.float32) / 32767.0
                    cache[k] = vec
                self._shard_cache[shard_id] = cache
            cache = self._shard_cache.get(shard_id, {})
            vec = cache.get(key)
            return vec
        except Exception:
            return None

    def _save_weave_index(self):
        try:
            data = {
                'index': self._weave_index,
                'layers': {str(k): list(v) for k,v in self._weave_layers.items()}
            }
            WEAVE_INDEX.write_text(json.dumps(data), encoding='utf-8')
        except Exception:
            pass

    def _log_ledger(self, key, action):
        try:
            entry = {'key': key, 'action': action, 'ts': __import__('time').time()}
            with LEDGER_PATH.open('a', encoding='utf-8') as f:
                f.write(json.dumps(entry) + '\n')
        except Exception:
            pass

    def _warm_up_vocab(self, target_count=10000):
        try:
            # Load recent keys from BitDrop store to warm weave
            if STORE_BD2.exists():
                sys_path = str(BASE / "scripts")
                import sys as _sys
                if sys_path not in _sys.path:
                    _sys.path.insert(0, sys_path)
                from bitdrop_persistence import load_lines_bitdrop
                lines = load_lines_bitdrop(STORE_BD2)
                count = 0
                for obj in lines:
                    key = obj.get('key')
                    if not key:
                        continue
                    if key not in self._weave_index:
                        layer_id = 0
                        self._weave_index[key] = {'layer': layer_id, 'neighbors': [], 'access_count': 1}
                        self._weave_layers.setdefault(layer_id, set()).add(key)
                    count += 1
                    if count >= target_count:
                        break
                self._save_weave_index()
        except Exception:
            pass

    def _load(self):
        # Sharded lazy load: do not load all vectors at startup
        # print("[EMBEDDINGS] _load start - sharded lazy mode")
        # Vectors will be loaded on demand from shards via index
        # Keep existing fallback for legacy JSONL migration
        if STORE_JSON.exists() and not INDEX_PATH.exists():
            # Migrate legacy store to shards if needed
            try:
                from scripts.shard_embed_store_v2 import main as shard_main
                # Not running auto-migrate here
                pass
            except Exception:
                pass
        # print("[EMBEDDINGS] _load complete - vectors lazy")

    def _migrate_to_bd2(self):
        try:
            # Quantize to int16 for storage
            entries = []
            for k,v in self._vectors.items():
                # Clip to [-1,1]
                v_clipped = np.clip(v, -1.0, 1.0)
                # Quantize to int16
                q = (v_clipped * 32767.0).astype(np.int16).tolist()
                entries.append({"key": k, "vec": q})
            # Write via BitDrop
            sys_path = str(BASE / "scripts")
            import sys as _sys
            if sys_path not in _sys.path:
                _sys.path.insert(0, sys_path)
            from bitdrop_persistence import save_line
            # Overwrite existing
            if STORE_BD2.exists():
                STORE_BD2.unlink()
            for e in entries:
                save_line(STORE_BD2, e)
            # Save meta
            meta = {"dim": DIM, "quant": "int16", "count": len(entries)}
            STORE_META.write_text(json.dumps(meta))
        except Exception:
            pass

    def _save_one(self, key, vec):
        # Quantize and append via BitDrop
        try:
            v_clipped = np.clip(vec, -1.0, 1.0)
            q = (v_clipped * 32767.0).astype(np.int16).tolist()
            obj = {"key": key, "vec": q}
            sys_path = str(BASE / "scripts")
            import sys as _sys
            if sys_path not in _sys.path:
                _sys.path.insert(0, sys_path)
            from bitdrop_persistence import save_line
            save_line(STORE_BD2, obj)
            # Update bloom filter for fast membership
            try:
                self._bloom.add(key)
            except Exception:
                pass
        except Exception:
            # Fallback to JSONL
            obj = {"key": key, "vec": vec.tolist()}
            with STORE_JSON.open('a', encoding='utf-8') as f:
                f.write(json.dumps(obj) + "\n")
            try:
                self._bloom.add(key)
            except Exception:
                pass

    def _touch(self, key):
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)
        # Evict oldest if over limit
        while len(self._vectors) > self._max_ram_items:
            oldest = self._access_order.pop(0)
            if oldest in self._vectors:
                del self._vectors[oldest]

    def _prefetch_cluster(self, key: str):
        """Prefetch hot neighbors from same cluster for cache efficiency"""
        try:
            if not hasattr(self, '_cluster_map') or not self._cluster_map:
                return
            cluster_id = self._cluster_map.get(key)
            if cluster_id is None:
                return
            # Find up to 4 hot neighbors in same cluster
            candidates = []
            for k, cid in self._cluster_map.items():
                if cid == cluster_id and k != key:
                    h = self._heat.get(k, 0.0)
                    if h > self._heat_hot_threshold:
                        candidates.append((h, k))
            candidates.sort(reverse=True)
            for _, k2 in candidates[:4]:
                if k2 not in self._vectors and k2 in self._index_map:
                    shard_id = self._index_map[k2]
                    vec = self._load_from_shard(shard_id, k2)
                    if vec is not None:
                        self._vectors[k2] = vec
                        self._touch(k2)
        except Exception:
            pass

    def get(self, text: str) -> np.ndarray:
        if not self._warm_up_done:
            self._warm_up_vocab(20000)
            self._warm_up_done = True
        key = text.strip().lower()
        # Bloom filter fast negative check
        if key not in self._bloom and key not in self._index_map and key not in self._weave_index:
            # Likely new key, skip expensive shard lookup
            # Fall through to init
            pass
        # Update heatmap on access
        self._update_heat(key, delta=0.1)
        # Skim layer: cold embeddings use fast approximation
        heat = self._heat.get(key, 0.0)
        if heat < self._skim_threshold and key not in self._vectors:
            self._skim_skips += 1
            # Return stable-seed approximation without I/O
            vec = _init_vector(text)
            return vec
        # Weave index lookup for fast routing
        meta = self._weave_index.get(key)
        if meta:
            meta['access_count'] = meta.get('access_count', 0) + 1
            # Promote layer on frequent access
            if meta['access_count'] % 10 == 0:
                old_layer = meta.get('layer', 0)
                new_layer = max(0, old_layer - 1)
                meta['layer'] = new_layer
                # Move between layers
                if old_layer in self._weave_layers:
                    self._weave_layers[old_layer].discard(key)
                self._weave_layers.setdefault(new_layer, set()).add(key)
            self._weave_index[key] = meta
            self._save_weave_index()
            self._log_ledger(key, 'weave_hit')
        if key in self._vectors:
            self._touch(key)
            # Opportunistic prefetch on hit
            self._prefetch_cluster(key)
            return self._vectors[key]
        # Try lazy load from shard index
        if key in self._index_map:
            shard_id = self._index_map[key]
            vec = self._load_from_shard(shard_id, key)
            if vec is not None:
                self._vectors[key] = vec
                self._touch(key)
                # Cluster-aware prefetch: load hot neighbors from same cluster
                self._prefetch_cluster(key)
                return vec
        # If weave index indicates key exists but not in RAM, try fast load via index
        if meta:
            # Attempt to keep vector hot
            pass
        vec = _init_vector(text)
        self._vectors[key] = vec
        self._touch(key)
        self._save_one(key, vec)
        # Register in weave lattice
        if key not in self._weave_index:
            layer_id = len(self._weave_layers)
            self._weave_index[key] = {'layer': layer_id, 'neighbors': [], 'access_count': 1}
            self._weave_layers.setdefault(layer_id, set()).add(key)
            self._save_weave_index()
        self._log_ledger(key, 'created')
        return vec

    def update(self, text: str, delta: np.ndarray):
        key = text.strip().lower()
        vec = self.get(text)
        # Heat-modulated learning: hot embeddings learn faster, cold learn slower
        heat = self._heat.get(key, 0.0)
        # Scale delta by heat with ReLU gating for cold overlay
        heat_gate = max(0.0, heat)  # ReLU
        scale = 0.5 + 1.5 * heat_gate  # 0.5 to 2.0 range
        delta_scaled = delta.astype(np.float32) * scale
        vec = vec + delta_scaled
        norm = np.linalg.norm(vec) + 1e-9
        vec = vec / norm
        self._vectors[key] = vec
        self._touch(key)
        # Boost heat on update
        self._update_heat(key, delta=0.2)
        self._save_one(key, vec)

    def cosine(self, a: str, b: str) -> float:
        va = self.get(a)
        vb = self.get(b)
        # Use self-caching matrix multiply for cheap CPU dot product
        try:
            from infrastructure.predictors.matrix_multiply_predictor import cached_matmul
            dot = cached_matmul(va, vb, op="*")
            if dot is not None:
                return float(np.dot(va, vb) if isinstance(dot, (list,)) else dot)
        except Exception:
            pass
        return float(np.dot(va, vb))

    def find_similar(self, query_key: str, k: int = 10) -> List[Tuple[str, float]]:
        """Cluster-restricted approximate nearest neighbors.
        Uses cluster map to limit candidate set to same cluster + hot neighbors,
        falling back to full scan if cluster map unavailable.
        """
        try:
            import numpy as np
            qvec = self.get(query_key)
            qnorm = np.linalg.norm(qvec) + 1e-9
            qvec = qvec / qnorm
            candidates = []
            # Restrict to same cluster if available
            if hasattr(self, '_cluster_map') and self._cluster_map:
                cid = self._cluster_map.get(query_key)
                if cid is not None:
                    # Build candidate pool from same cluster
                    for k2, c2 in self._cluster_map.items():
                        if c2 == cid and k2 != query_key:
                            candidates.append(k2)
                    # Also add hot keys from same layer for recall
                    layer = self._weave_index.get(query_key, {}).get('layer', 0)
                    for k2 in self._weave_layers.get(layer, set()):
                        if k2 != query_key and k2 not in candidates:
                            candidates.append(k2)
            else:
                # Fallback: all keys with heat > cold threshold
                candidates = [k for k, h in self._heat.items() if h >= self._heat_cold_threshold and k != query_key]
            # Limit candidate set size for speed
            if len(candidates) > 2000:
                # Sample by heat
                candidates = sorted(candidates, key=lambda k: self._heat.get(k, 0.0), reverse=True)[:2000]
            results = []
            for k2 in candidates:
                v2 = self.get(k2)
                # Normalize
                n2 = np.linalg.norm(v2) + 1e-9
                v2n = v2 / n2
                sim = float(np.dot(qvec, v2n))
                results.append((k2, sim))
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:k]
        except Exception:
            return []

store = EmbeddingStore()

def embed(text: str) -> np.ndarray:
    return store.get(text)

def update_embedding(text: str, delta: np.ndarray):
    store.update(text, delta)
