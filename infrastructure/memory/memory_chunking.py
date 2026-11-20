"""
Production-grade chunking, grouping, clustering and skimming for AI Memory
Improves retrieval speed and reduces redundancy in Delta-backed vector memory.
"""
from __future__ import annotations
import threading
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict

class ChunkingEngine:
    """Split long token/embedding sequences into overlapping chunks."""
    def __init__(self, chunk_size: int = 256, overlap: int = 64):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.lock = threading.RLock()

    def chunk_vector(self, vid: str, vector: np.ndarray) -> List[Tuple[str, np.ndarray]]:
        with self.lock:
            v = np.asarray(vector, dtype=np.float32)
            if v.size <= self.chunk_size:
                return [(vid, v)]
            chunks = []
            step = self.chunk_size - self.overlap
            for i in range(0, v.size, step):
                chunk = v[i:i+self.chunk_size]
                cid = f"{vid}::c{i//step}"
                chunks.append((cid, chunk))
                if i + self.chunk_size >= v.size:
                    break
            return chunks

class GroupingEngine:
    """Group memories by semantic similarity using coarse clustering."""
    def __init__(self, dim: int = 256):
        self.dim = dim
        self.groups: Dict[str, List[str]] = defaultdict(list)
        self.lock = threading.RLock()

    def group(self, vids: List[str], embeddings: Dict[str, np.ndarray], n_groups: int = 32) -> Dict[str, List[str]]:
        with self.lock:
            if not vids:
                return {}
            # Simple cosine similarity based k-means init
            X = np.stack([embeddings[vid] for vid in vids])
            # Normalize
            norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
            X_norm = X / norms
            # Initialize centroids randomly
            rng = np.random.default_rng(0)
            centroids = X_norm[rng.choice(len(vids), min(n_groups, len(vids)), replace=False)]
            for _ in range(10):
                dists = 1 - X_norm @ centroids.T
                labels = np.argmin(dists, axis=1)
                new_centroids = []
                for k in range(len(centroids)):
                    members = X_norm[labels == k]
                    if len(members) == 0:
                        new_centroids.append(centroids[k])
                    else:
                        new_centroids.append(members.mean(axis=0))
                centroids = np.vstack(new_centroids)
            groups = defaultdict(list)
            for vid, lab in zip(vids, labels):
                groups[f"g{lab}"].append(vid)
            self.groups = dict(groups)
            return dict(groups)

class SkimmingEngine:
    """Fast coarse filter for retrieval: coarse → medium → fine."""
    def __init__(self, dim: int = 256):
        self.dim = dim
        self.coarse_codes: Dict[str, int] = {}
        self.lock = threading.RLock()

    def encode_coarse(self, vector: np.ndarray) -> int:
        v = np.asarray(vector, dtype=np.float32)
        # 8-bit per dim hash
        h = 0
        for i in range(min(64, v.size)):
            h = (h * 31 + int(abs(v[i]) * 1000)) & 0xffffffff
        return h % 1024

    def skim(self, query_vec: np.ndarray, candidates: List[str], candidate_embs: Dict[str, np.ndarray], top_k: int = 100) -> List[str]:
        with self.lock:
            q_code = self.encode_coarse(query_vec)
            # First pass: coarse code match
            filtered = []
            for vid in candidates:
                code = self.coarse_codes.get(vid)
                if code is None:
                    code = self.encode_coarse(candidate_embs.get(vid, np.zeros(self.dim)))
                    self.coarse_codes[vid] = code
                if abs(code - q_code) < 256:  # coarse band
                    filtered.append(vid)
            if not filtered:
                filtered = candidates[:min(top_k*2, len(candidates))]
            # Second pass: cheap dot product
            q = query_vec / (np.linalg.norm(query_vec) + 1e-9)
            scores = []
            for vid in filtered:
                e = candidate_embs.get(vid)
                if e is None:
                    continue
                e_n = e / (np.linalg.norm(e) + 1e-9)
                scores.append((vid, float(q @ e_n)))
            scores.sort(key=lambda x: x[1], reverse=True)
            return [vid for vid, _ in scores[:top_k]]

class MemoryChunkingOrchestrator:
    """Combines chunking, grouping, clustering and skimming."""
    def __init__(self, dim: int = 256):
        self.chunking = ChunkingEngine()
        self.grouping = GroupingEngine(dim)
        self.skimming = SkimmingEngine(dim)
        self.lock = threading.RLock()

    def ingest(self, vid: str, vector: np.ndarray) -> List[str]:
        with self.lock:
            chunks = self.chunking.chunk_vector(vid, vector)
            # Return chunk ids
            return [cid for cid, _ in chunks]

    def skim_and_group(self, query_vec: np.ndarray, candidate_ids: List[str], embeddings: Dict[str, np.ndarray], k: int = 50) -> Tuple[List[str], Dict[str, List[str]]]:
        with self.lock:
            skimmed = self.skimming.skim(query_vec, candidate_ids, embeddings, top_k=max(k*4, 200))
            groups = self.grouping.group(skimmed, embeddings, n_groups=max(8, k//4))
            # Return top groups flattened
            top_vids = []
            for gids in groups.values():
                top_vids.extend(gids)
                if len(top_vids) >= k:
                    break
            return top_vids[:k], groups
