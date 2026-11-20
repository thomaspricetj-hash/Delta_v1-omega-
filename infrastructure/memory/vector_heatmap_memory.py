"""
Production-grade Vector Layered Heatmap Memory with Cold Overlay
Multilayered Index + Multilayered Ledger + Chunking/Grouping/Clustering + Bloom Filter

Features:
- Delta-backed vector storage with reversible deltas
- Multilayered index: coarse/medium/fine ANN
- Multilayered graph for semantic links
- Multilayered ledger with DAX-style journaling, BitDrop-v2 compression ready
- Multilayered heatmap with cold overlay, diffusion, decay
- Cold overlay: explicit cold region tracking and gating to prevent hot-spot bias
- Chunking/Grouping/Clustering for shards with Bloom-filter fast negative checks
- Thread-safe, Windows CPU friendly
"""
from __future__ import annotations
import threading
import hashlib
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

from .vector_store import DeltaVector, MultilayeredIndex, MultilayeredGraph, MultilayeredLedger
from .tree_vector_store import TreeDeltaVectorStore
from ..heatmap.grid_heatmap import MultilayeredHeatmap

try:
    from sklearn.cluster import MiniBatchKMeans
    SKLEARN_CLUSTER = True
except Exception:
    SKLEARN_CLUSTER = False
    MiniBatchKMeans = None

# ---------- Bloom Filter ----------
class BloomFilter:
    def __init__(self, size: int = 1 << 22, hashes: int = 3):
        self.size = size
        self.hashes = hashes
        self.bits = bytearray((size + 7) // 8)

    def _hashes(self, key: str):
        h = hashlib.sha256(key.encode()).digest()
        for i in range(self.hashes):
            yield int.from_bytes(h[i*4:(i+1)*4], 'big') % self.size

    def add(self, key: str):
        for h in self._hashes(key):
            self.bits[h // 8] |= 1 << (h % 8)

    def __contains__(self, key: str) -> bool:
        return all((self.bits[h // 8] >> (h % 8)) & 1 for h in self._hashes(key))

# ---------- Cold Overlay ----------
class ColdOverlay:
    """
    Tracks cold regions to balance heatmap activation.
    Cold overlay increases decay for over-heated cells and boosts under-used cells.
    """
    def __init__(self, layers: int = 4, width: int = 64, height: int = 64):
        self.layers = layers
        self.width = width
        self.height = height
        self.cold_map = np.zeros((layers, height, width), dtype=np.float32)
        self.lock = threading.RLock()
        self.usage_counter = np.zeros((layers, height, width), dtype=np.int32)

    def register_access(self, layer: int, x: int, y: int):
        with self.lock:
            self.usage_counter[layer, y, x] += 1

    def update_cold(self, heatmap: MultilayeredHeatmap):
        with self.lock:
            for l in range(self.layers):
                hm = heatmap.maps.get(l)
                if hm is None:
                    continue
                heat = hm.heat
                # Normalize usage
                usage = self.usage_counter[l]
                # Cold score = inverse of heat * usage
                # High heat + high usage -> less cold
                # Low heat + low usage -> more cold -> boost
                cold = 1.0 / (1.0 + heat + 1e-6) * (1.0 / (1.0 + usage))
                self.cold_map[l] = cold.astype(np.float32)

    def get_cold_boost(self, layer: int, x: int, y: int) -> float:
        with self.lock:
            return float(self.cold_map[layer, y, x])

    def reset_usage(self):
        with self.lock:
            self.usage_counter.fill(0)

# ---------- Chunking / Grouping / Clustering ----------
class VectorChunker:
    def __init__(self, chunk_size: int = 512, n_clusters: int = 256):
        self.chunk_size = chunk_size
        self.n_clusters = n_clusters
        self.bloom = BloomFilter()
        self.cluster_model = None
        self.lock = threading.RLock()

    def chunk_id(self, vid: str) -> str:
        # Deterministic chunk bucket
        h = hashlib.sha256(vid.encode()).hexdigest()
        return f"chunk_{int(h, 16) % self.chunk_size}"

    def group_id(self, vid: str) -> str:
        # Coarser grouping
        return f"group_{hashlib.sha256(vid.encode()).hexdigest()[:8]}"

    def cluster_assign(self, vector: np.ndarray) -> int:
        with self.lock:
            if not SKLEARN_CLUSTER or self.cluster_model is None:
                return 0
            # MiniBatchKMeans expects 2D
            v = vector.reshape(1, -1)
            label = self.cluster_model.predict(v)[0]
            return int(label)

    def fit_clusters(self, vectors: List[np.ndarray]):
        if not SKLEARN_CLUSTER:
            return
        with self.lock:
            if not vectors:
                return
            X = np.vstack([v for v in vectors])
            n_clusters = min(self.n_clusters, max(1, len(vectors)//10))
            self.cluster_model = MiniBatchKMeans(n_clusters=n_clusters, batch_size=256, n_init=1, random_state=0)
            self.cluster_model.fit(X)

    def ingest(self, vid: str):
        self.bloom.add(vid)
        chunk = self.chunk_id(vid)
        self.bloom.add(chunk)
        return chunk

# ---------- Unified Layered Heatmap Memory ----------
class VectorLayeredHeatmapMemory:
    def __init__(self, dim: int = 256, ledger_path: Optional[Path] = None, memory_root: Optional[Path] = None):
        self.dim = dim
        self.lock = threading.RLock()
        self.store = TreeDeltaVectorStore(dim=dim, capacity_per_node=5000, dir_path=str(Path(memory_root) / "vecmem_tree" if memory_root else "runtime/memory/vecmem_tree"))
        self.index = MultilayeredIndex(dim=dim)
        self.graph = MultilayeredGraph()
        self.ledger = MultilayeredLedger(path=ledger_path)
        self.base_layers = 4
        self.heatmap = MultilayeredHeatmap(layers=self.base_layers, width=64, height=64)
        self.cold_overlay = ColdOverlay(layers=self.base_layers, width=64, height=64)
        self.chunker = VectorChunker(chunk_size=512, n_clusters=256)
        self.bloom = BloomFilter(size=1 << 22, hashes=3)
        self.memory_root = Path(memory_root) if memory_root else Path("runtime/memory")
        self._load()
        # Ensure layers are expanded if needed after load
        self._ensure_layers(len(self.store.vectors))

    def _ensure_layers(self, n_vectors: int):
        # Dynamic layer expansion to keep per-layer density low and access fast
        # Target ~5000 vectors per layer
        target_per_layer = 5000
        desired_layers = max(self.base_layers, (n_vectors + target_per_layer - 1) // target_per_layer)
        current_layers = len(self.heatmap.maps)
        if desired_layers <= current_layers:
            return
        # Expand heatmap
        for l in range(current_layers, desired_layers):
            self.heatmap.maps[l] = self.heatmap.maps[0].__class__(64, 64)
            self.heatmap.layer_weights = np.pad(self.heatmap.layer_weights, (0, 1), constant_values=1.0)
        # Expand cold overlay
        old_layers = self.cold_overlay.layers
        new_layers = desired_layers
        if new_layers > old_layers:
            new_cold = np.zeros((new_layers, 64, 64), dtype=np.float32)
            new_usage = np.zeros((new_layers, 64, 64), dtype=np.int32)
            new_cold[:old_layers] = self.cold_overlay.cold_map
            new_usage[:old_layers] = self.cold_overlay.usage_counter
            self.cold_overlay.cold_map = new_cold
            self.cold_overlay.usage_counter = new_usage
            self.cold_overlay.layers = new_layers

    def _load(self):
        base = self.memory_root / "vecmem_heatmap"
        vectors_path = base / "vectors.json"
        if vectors_path.exists():
            self.store.load(vectors_path)
            for vid, dv in self.store.vectors.items():
                self.index.upsert(vid, dv.materialize())
                self.bloom.add(vid)
                self.chunker.bloom.add(vid)

    def save(self, base: Optional[Path] = None):
        with self.lock:
            base = Path(base) if base else self.memory_root / "vecmem_heatmap"
            base.mkdir(parents=True, exist_ok=True)
            self.store.save()
            # Save heatmap state
            heat_data = {}
            for l, hm in self.heatmap.maps.items():
                heat_data[str(l)] = {
                    "heat": hm.heat.tolist(),
                    "age": hm.age.tolist(),
                }
            with open(base / "heatmap.json", "w", encoding="utf-8") as f:
                json.dump(heat_data, f)
            # Save cold overlay
            cold_data = {
                "cold_map": self.cold_overlay.cold_map.tolist(),
                "usage": self.cold_overlay.usage_counter.tolist(),
            }
            with open(base / "cold_overlay.json", "w", encoding="utf-8") as f:
                json.dump(cold_data, f)

    def add(self, vid: str, vector: np.ndarray, metadata: Optional[dict] = None):
        with self.lock:
            v = np.asarray(vector, dtype=np.float32)
            if v.shape[0] != self.dim:
                raise ValueError("dim mismatch")
            # Bloom fast path
            if vid in self.bloom:
                # Already present, treat as update
                pass
            self.store.add(vid, v)
            self.index.upsert(vid, v)
            self.bloom.add(vid)
            self.chunker.ingest(vid)
            # Heatmap activation
            act_val = float(np.linalg.norm(v) / (self.dim ** 0.5))
            # Layer 0 for semantic, layer 1 for magnitude, layer 2 for frequency
            self.heatmap.add_activation(0, vid, act_val, radius=1)
            self.heatmap.add_activation(1, vid, act_val * 0.5, radius=1)
            # Cold overlay update
            # Deterministic coord for usage tracking
            h = hashlib.sha256(vid.encode()).digest()
            x = int.from_bytes(h[:4], 'big') % 64
            y = int.from_bytes(h[4:8], 'big') % 64
            self.cold_overlay.register_access(0, x, y)
            # Ledger
            payload = json.dumps({"vid": vid, "dim": int(v.shape[0]), "chunk": self.chunker.chunk_id(vid)}).encode()
            self.ledger.append("add", vid, payload, layer=2)
            # Dynamic layer expansion to keep per-layer density low
            self._ensure_layers(len(self.store.vectors))
            # Periodic decay
            if np.random.rand() < 0.01:
                self.heatmap.decay_all()
                self.cold_overlay.update_cold(self.heatmap)

    def update(self, vid: str, delta: np.ndarray):
        with self.lock:
            if vid not in self.bloom:
                raise KeyError(vid)
            self.store.update(vid, delta)
            v = self.store.get(vid)
            self.index.upsert(vid, v)
            # Update heatmap with delta magnitude
            act_val = float(np.linalg.norm(delta))
            self.heatmap.add_activation(0, vid, act_val * 0.1, radius=1)
            payload = json.dumps({"vid": vid, "delta_norm": float(np.linalg.norm(delta))}).encode()
            self.ledger.append("update", vid, payload, layer=2)

    def query(self, query_vec: np.ndarray, k: int = 10) -> List[Tuple[str, float, float]]:
        with self.lock:
            # ANN search
            ann = self.index.search(query_vec, k * 4)
            # Heat and cold gating
            results = []
            for vid, dist in ann:
                if vid not in self.bloom:
                    continue
                # Get heat
                heat = self.heatmap.get_heat(0, vid)
                # Cold boost
                h = hashlib.sha256(vid.encode()).digest()
                x = int.from_bytes(h[:4], 'big') % 64
                y = int.from_bytes(h[4:8], 'big') % 64
                cold_boost = self.cold_overlay.get_cold_boost(0, x, y)
                # Score: distance inverse + heat + cold
                score = 1.0 / (1.0 + dist) * (1.0 + 0.2 * heat) * (1.0 + 0.1 * cold_boost)
                results.append((vid, score, heat))
            results.sort(key=lambda t: t[1], reverse=True)
            return results[:k]

    def link(self, src: str, dst: str, weight: float = 0.9, layer: int = 1):
        with self.lock:
            self.graph.add_edge(src, dst, weight, layer)
            # Graph-heat coupling: propagate heat along the edge
            try:
                src_heat = self.heatmap.get_heat(0, src)
                dst_heat = self.heatmap.get_heat(0, dst)
                # Boost both nodes proportionally to edge weight
                boost = 0.05 * weight * (1.0 + min(src_heat, dst_heat))
                # Re-activate via heatmap add_activation with small value
                self.heatmap.add_activation(2, src, boost, radius=1)
                self.heatmap.add_activation(2, dst, boost, radius=1)
            except Exception:
                pass
            payload = json.dumps({"src": src, "dst": dst, "w": weight}).encode()
            self.ledger.append("link", src, payload, layer=layer)

    def health(self) -> dict:
        with self.lock:
            return {
                "vectors": len(self.store.vectors),
                "index_entries": sum(len(v) for v in self.index.id_to_vec.values()) if hasattr(self.index, 'id_to_vec') else 0,
                "graph_edges": len(self.graph.edges),
                "heatmap_layers": len(self.heatmap.maps),
                "cold_active": True,
                "chunker_active": True,
                "bloom_size": self.bloom.size,
            }

# Helper factory
def create_vector_heatmap_memory(dim: int = 256, memory_root: Optional[str] = None):
    return VectorLayeredHeatmapMemory(dim=dim, memory_root=memory_root)
