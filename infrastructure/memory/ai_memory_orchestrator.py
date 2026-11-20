"""
Production-grade AI Memory Orchestrator for TRANSLATION_OS_V1
Unified interface for Delta-backed Vector Memory + Bio Memory + Reasoning Memory
with Bloom-accelerated membership, multilayered index/graph/ledger, and ML-guided
retrieval. Thread-safe, persistent, and Windows CPU-friendly.
"""
from __future__ import annotations
import threading
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np
import json

from .vector_store import DeltaVectorMemory, BloomFilter
from .vector_heatmap_memory import VectorLayeredHeatmapMemory
from .heart_memory import BioMemory
from .reasoning_memory import ReasoningMemory
from .memory_controller import MemoryController
from .memory_chunking import MemoryChunkingOrchestrator
from .family_tree import FamilyTree
try:
    from ..heatmap.grid_heatmap import MultilayeredHeatmap
    HEATMAP_AVAILABLE = True
except Exception:
    HEATMAP_AVAILABLE = False
    MultilayeredHeatmap = None

log = logging.getLogger(__name__)

class AIMemoryOrchestrator:
    """
    Single entry point for all memory operations in the training/inference pipeline.
    Handles delta compression, bio consolidation, reasoning recall, and persistence.
    """
    def __init__(self, dim: int = 256, ledger_path: Optional[Path] = None, memory_root: Optional[Path] = None):
        self.dim = dim
        self.lock = threading.RLock()
        self.ledger_path = ledger_path
        self.memory_root = Path(memory_root) if memory_root else Path("runtime/memory")
        self.memory_root.mkdir(parents=True, exist_ok=True)

        # Use Vector Layered Heatmap Memory with cold overlay, multilayered index/ledger, chunking/clustering + Bloom
        self.vector_mem = VectorLayeredHeatmapMemory(dim=dim, ledger_path=ledger_path, memory_root=self.memory_root)
        # Compatibility shim for existing code expecting DeltaVectorMemory API
        self.vector_mem.add_memory = self.vector_mem.add
        self.vector_mem.update_memory = self.vector_mem.update
        self.vector_mem.link_memories = self.vector_mem.link
        # Expose store/index/graph/ledger attributes used by orchestrator
        # Already present via VectorLayeredHeatmapMemory
        self.bio_mem = BioMemory(dim=dim, vector_mem=self.vector_mem)
        self.reasoning_mem = ReasoningMemory(bio_mem=self.bio_mem, vector_mem=self.vector_mem)
        self.controller = MemoryController()
        self.chunk_orch = MemoryChunkingOrchestrator(dim=dim)
        self.family_tree = FamilyTree()
        # Use heatmap from vector memory for spatial gating
        self.heatmap = self.vector_mem.heatmap

        # Separate Bloom for fast negative checks on high-level IDs
        self.id_bloom = BloomFilter(size=1 << 22, hashes=4)

        self._load()

    # ---------- Persistence ----------
    def _load(self):
        # VectorLayeredHeatmapMemory loads itself from memory_root/vecmem_heatmap on init
        # Repopulate top-level bloom from current store
        try:
            store = self.vector_mem.store
            # TreeDeltaVectorStore uses in_memory dict, not vectors
            vecs = getattr(store, 'vectors', None)
            if vecs is None:
                vecs = getattr(store, 'in_memory', {})
            for vid in vecs.keys():
                self.id_bloom.add(vid)
            log.info(f"AIMemory loaded {len(vecs)} vectors")
        except Exception as e:
            log.warning(f"AIMemory load skipped: {e}")

    def save(self):
        with self.lock:
            # Save to vecmem_heatmap location used by VectorLayeredHeatmapMemory
            self.vector_mem.save(self.memory_root)
            # Persist bio state if needed
            store = self.vector_mem.store
            vecs = getattr(store, 'vectors', None)
            if vecs is None:
                vecs = getattr(store, 'in_memory', {})
            log.info(f"AIMemory saved {len(vecs)} vectors")

    # ---------- Core API ----------
    def store_experience(self, vid: str, vector: np.ndarray, token_ids: List[int], context: str = "") -> None:
        with self.lock:
            v = np.asarray(vector, dtype=np.float32)
            if v.shape[0] != self.dim:
                raise ValueError("dim mismatch")
            # Chunk long vectors for better granularity
            chunk_ids = self.chunk_orch.ingest(vid, v)
            for cid in chunk_ids:
                if cid != vid:
                    self.id_bloom.add(cid)
            # Family tree lineage: link to most similar recent ancestor
            self._link_family(vid, v)
            # Capture most recent vector before insertion for temporal linking
            recent_ids_before = list(self.vector_mem.store.vectors.keys())
            prev_vid = recent_ids_before[-1] if recent_ids_before else None
            self.bio_mem.store_experience(vid, v, token_ids, context)
            self.id_bloom.add(vid)
            # Update multilayered heatmap with activation magnitude
            if self.heatmap is not None:
                act_val = float(np.linalg.norm(v) / (self.dim ** 0.5))
                self.heatmap.add_activation(0, vid, act_val, radius=1)
                # Decay heatmaps periodically
                if np.random.rand() < 0.01:
                    self.heatmap.decay_all()
            # Create temporal graph link to previous experience
            if prev_vid and prev_vid != vid:
                self.vector_mem.link_memories(prev_vid, vid, weight=0.7, layer=2)
            self.vector_mem.ledger.append("bio_store", vid, json.dumps({"context": context}).encode(), layer=1)

    def recall(self, query_vec: np.ndarray, k: int = 10) -> List[Dict]:
        with self.lock:
            q = np.asarray(query_vec, dtype=np.float32)
            # Try cluster-restricted candidate generation via EmbeddingStore for speed
            candidate_pool = None
            try:
                # Find a seed candidate via ANN to anchor cluster search
                if self.vector_mem and hasattr(self.vector_mem, 'index'):
                    ann_candidates = self.vector_mem.index.search(q, k=4)
                    if ann_candidates:
                        seed_vid = ann_candidates[0][0]
                        # Use EmbeddingStore cluster-restricted similarity
                        from infrastructure.embeddings.embeddings import store as embed_store
                        similar = embed_store.find_similar(seed_vid, k=k*4)
                        if similar:
                            candidate_pool = [vid for vid, _ in similar]
            except Exception:
                candidate_pool = None
            hits = self.reasoning_mem.recall(q, k=k*2, candidate_pool=candidate_pool)
            # Heatmap gating: filter candidates by spatial heat
            if self.heatmap is not None:
                candidate_ids = [h.vid for h in hits]
                filtered_ids = self.heatmap.filter_candidates_by_heat(candidate_ids, layer=0, top_p=0.5)
                # Re-score with heat boost
                heat_boost = {}
                for vid in filtered_ids:
                    hval = self.heatmap.get_heat(0, vid)
                    heat_boost[vid] = 1.0 + 0.2 * min(2.0, hval)
            else:
                heat_boost = {h.vid: 1.0 for h in hits}
            # Family tree boost: increase score for nodes with close ancestors
            boosted = []
            for h in hits:
                boost = 1.0
                ancestors = self.family_tree.get_ancestors(h.vid, max_depth=3)
                if ancestors:
                    boost += 0.1 * len(ancestors)
                descendants = self.family_tree.get_descendants(h.vid)
                if descendants:
                    boost += 0.05 * min(5, len(descendants))
                # Heatmap boost
                hb = heat_boost.get(h.vid, 1.0)
                score = h.score * boost * hb
                boosted.append((h, score))
            boosted.sort(key=lambda x: x[1], reverse=True)
            top = boosted[:k]
            return [
                {
                    "vid": h.vid,
                    "score": float(score),
                    "heart": float(h.heart_score),
                    "muscle": float(h.muscle_score),
                    "graph": float(h.graph_score),
                    "family_boost": float(score / (h.score + 1e-9)),
                }
                for h, score in top
            ]

    def augment_context(self, query_vec: np.ndarray, k: int = 5) -> List[str]:
        with self.lock:
            # Cluster-restricted candidate generation for augment_context
            candidate_pool = None
            try:
                if self.vector_mem and hasattr(self.vector_mem, 'index'):
                    ann_candidates = self.vector_mem.index.search(query_vec, k=4)
                    if ann_candidates:
                        seed_vid = ann_candidates[0][0]
                        from infrastructure.embeddings.embeddings import store as embed_store
                        similar = embed_store.find_similar(seed_vid, k=k*4)
                        if similar:
                            candidate_pool = [vid for vid, _ in similar]
            except Exception:
                candidate_pool = None
            # Use reasoning memory with candidate pool
            hits = self.reasoning_mem.recall(query_vec, k=k, candidate_pool=candidate_pool)
            return [h.vid for h in hits[:k]]

    def update_memory(self, vid: str, delta: np.ndarray) -> None:
        with self.lock:
            if vid not in self.id_bloom:
                # Bloom says definitely not present
                raise KeyError(vid)
            self.vector_mem.update_memory(vid, delta)

    def link_memories(self, src: str, dst: str, weight: float = 0.9, layer: int = 1) -> None:
        with self.lock:
            self.vector_mem.link_memories(src, dst, weight, layer)

    # ---------- Maintenance ----------
    def maintenance_step(self) -> dict:
        with self.lock:
            # Metrics
            n_vec = len(self.vector_mem.store.vectors)
            if n_vec == 0:
                avg_chain = 0.0
            else:
                chains = [len(v.deltas) for v in self.vector_mem.store.vectors.values()]
                avg_chain = float(np.mean(chains))
            # Simple recall probe: average reasoning score on recent vectors
            recall_score = 0.5
            try:
                # probe first few vectors
                sample_ids = list(self.vector_mem.store.vectors.keys())[:min(10, n_vec)]
                scores = []
                for vid in sample_ids:
                    v = self.vector_mem.store.get(vid)
                    hits = self.reasoning_mem.recall(v, k=3)
                    if hits:
                        scores.append(max(h.score for h in hits))
                if scores:
                    recall_score = float(np.mean(scores))
            except Exception:
                pass
            # Normalize memory size to [0,1] heuristic
            memory_size_norm = min(1.0, n_vec / 100000.0)

            compress_aggr, decay_adj, recall_u = self.controller.compute(avg_chain, recall_score, memory_size_norm)

            # Apply controls
            # Adaptive compress threshold
            base_threshold = 8
            thresh = self.controller.recommend_compress_threshold(base_threshold, compress_aggr)
            # Simple policy: compress if avg_chain >= thresh
            if avg_chain >= thresh:
                self.vector_mem.compress()
            # Adaptive heart decay
            base_decay = 0.995
            decay = self.controller.recommend_decay(base_decay, decay_adj)
            self.bio_mem.heart.consolidate(decay=decay)
            # Decay heatmaps for spatial forgetting
            if self.heatmap is not None:
                self.heatmap.decay_all()
            # LRU pruning to bound memory growth for multi-million step runs
            max_vectors = 200_000
            if len(self.vector_mem.store.vectors) > max_vectors:
                # Prune oldest 10% by insertion order
                ids = list(self.vector_mem.store.vectors.keys())
                prune_n = max(1, int(len(ids) * 0.1))
                for old_id in ids[:prune_n]:
                    try:
                        del self.vector_mem.store.vectors[old_id]
                        self.id_bloom.add(old_id)  # keep bloom for negative check, no removal
                    except Exception:
                        pass

            # Adaptive reasoning blend weights via recall PID
            # Map recall_u to weight shift
            w_shift = recall_u * 0.05
            # Keep weights normalized
            try:
                # Slightly increase heart/muscle weight if recall low
                self.reasoning_mem.w_heart = max(0.1, min(0.6, self.reasoning_mem.w_heart + w_shift))
                self.reasoning_mem.w_muscle = max(0.1, min(0.6, self.reasoning_mem.w_muscle + w_shift))
                # Renormalize
                total = self.reasoning_mem.w_heart + self.reasoning_mem.w_muscle + self.reasoning_mem.w_graph + self.reasoning_mem.w_ann
                self.reasoning_mem.w_heart /= total
                self.reasoning_mem.w_muscle /= total
                self.reasoning_mem.w_graph /= total
                self.reasoning_mem.w_ann /= total
            except Exception:
                pass

            log.debug(f"Memory maintenance: avg_chain={avg_chain:.2f}, recall={recall_score:.3f}, compress={compress_aggr:.3f}, decay={decay:.4f}")
            return {
                "avg_chain_len": avg_chain,
                "recall_score": recall_score,
                "compress_aggressiveness": compress_aggr,
                "heart_decay": decay,
                "memory_size_norm": memory_size_norm,
            }

    def compress(self) -> None:
        with self.lock:
            self.vector_mem.compress()
            self.bio_mem.heart.consolidate()
            log.debug("AIMemory compression + heart consolidation executed")

    def _link_family(self, vid: str, vector: np.ndarray):
        # Simple heuristic: link to most recent vector with high similarity
        recent_ids = list(self.vector_mem.store.vectors.keys())[-50:]
        best_sim = -1.0
        best_parent = None
        v_norm = np.linalg.norm(vector) + 1e-9
        for pid in recent_ids:
            if pid == vid:
                continue
            try:
                pv = self.vector_mem.store.get(pid)
                sim = float(np.dot(vector, pv) / (v_norm * (np.linalg.norm(pv)+1e-9)))
                if sim > best_sim:
                    best_sim = sim
                    best_parent = pid
            except Exception:
                continue
        if best_parent and best_sim > 0.75:
            self.family_tree.add_node(vid, parent_id=best_parent, embedding=vector, metadata={"sim": best_sim})
        else:
            self.family_tree.add_node(vid, parent_id=None, embedding=vector)

    def health_check(self) -> Dict[str, any]:
        with self.lock:
            return {
                "vectors": len(self.vector_mem.store.vectors),
                "index_entries": len(self.vector_mem.index.id_to_vec),
                "graph_edges": len(self.vector_mem.graph.edges),
                "ledger_entries": len(self.vector_mem.ledger.entries),
                "family_nodes": len(self.family_tree.nodes),
                "bloom_size": self.id_bloom.size,
            }

# ---------- Factory ----------
def create_orchestrator(dim: int = 256, memory_root: str = "runtime/memory") -> AIMemoryOrchestrator:
    return AIMemoryOrchestrator(
        dim=dim,
        ledger_path=Path(memory_root) / "vecmem_ledger.jsonl",
        memory_root=Path(memory_root)
    )

log.info("AIMemoryOrchestrator loaded")
