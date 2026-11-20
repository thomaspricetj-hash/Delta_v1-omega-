"""
Production-grade Reasoning Memory for TRANSLATION_OS_V1

Blends Heart rhythmic, Muscle procedural, and DeltaVector graph evidence
into a unified reasoning retrieval signal for WeaveFormer context augmentation.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict
import numpy as np
import threading

@dataclass
class ReasoningHit:
    vid: str
    score: float
    heart_score: float
    muscle_score: float
    graph_score: float
    source: str

class ReasoningMemory:
    """
    Unified reasoning recall that blends:
    - Heart rhythmic salience + phase coherence
    - Muscle procedural similarity + strength
    - Graph evidence propagation
    - DeltaVector ANN similarity
    """
    def __init__(self, bio_mem, vector_mem):
        self.bio_mem = bio_mem
        self.vector_mem = vector_mem
        self.lock = threading.RLock()
        # Blend weights - tunable
        self.w_heart = 0.35
        self.w_muscle = 0.35
        self.w_graph = 0.20
        self.w_ann = 0.10

    def _ann_score(self, query_vec: np.ndarray, vid: str) -> float:
        if self.vector_mem is None:
            return 0.0
        try:
            v = self.vector_mem.store.get(vid)
            q = np.asarray(query_vec, dtype=np.float32)
            sim = float(np.dot(q, v) / (np.linalg.norm(q) * np.linalg.norm(v) + 1e-9))
            return (sim + 1.0) / 2.0
        except Exception:
            return 0.0

    def _graph_score(self, vid: str, query_vec: np.ndarray) -> float:
        if self.vector_mem is None:
            return 0.0
        # Propagate evidence from neighbors
        scores = self.vector_mem.graph.propagate_evidence(vid, steps=2, decay=0.7)
        # Normalize
        if not scores:
            return 0.0
        max_s = max(scores.values())
        return scores.get(vid, 0.0) / (max_s + 1e-9)

    def recall(self, query_vec: np.ndarray, k: int = 10, candidate_pool: List[str] | None = None) -> List[ReasoningHit]:
        with self.lock:
            # Get candidates from ANN if pool not provided
            if candidate_pool is None:
                ann_candidates = self.vector_mem.index.search(query_vec, k * 4) if self.vector_mem else []
                candidate_ids = [vid for vid, _ in ann_candidates]
            else:
                candidate_ids = candidate_pool

            # Pre-compute heart and muscle scores for candidates to avoid zero lookups
            heart_hits = {}
            muscle_hits = {}
            if self.bio_mem:
                # Fallback direct scoring per candidate for robustness
                heart_mem = self.bio_mem.heart.memories
                muscle_mem = self.bio_mem.muscle
                q_norm = np.linalg.norm(query_vec) + 1e-9
                # Phase for current time
                phase_now = self.bio_mem.heart._current_phase()
                for vid in candidate_ids:
                    # Heart score
                    if vid in heart_mem:
                        mem = heart_mem[vid]
                        v = mem["vector"]
                        sim = float(np.dot(query_vec, v) / (q_norm * (np.linalg.norm(v) + 1e-9)))
                        phase_diff = abs(phase_now - mem["phase"])
                        phase_coherence = 1.0 - min(phase_diff, 1 - phase_diff) * 2
                        h_score = sim * mem["e_weight"] * (0.8 + 0.2 * phase_coherence)
                        heart_hits[vid] = h_score
                    # Muscle score
                    if vid in muscle_mem.skills:
                        emb = muscle_mem.embeddings.get(vid)
                        if emb is not None:
                            sim = float(np.dot(query_vec, emb) / (q_norm * (np.linalg.norm(emb) + 1e-9)))
                            strength = muscle_mem.skills[vid].strength
                            muscle_hits[vid] = sim * strength
                # Also keep top-k rhythmic query as a supplement
                heart_res = self.bio_mem.heart.rhythmic_query(query_vec, k * 4)
                for vid, score in heart_res:
                    heart_hits.setdefault(vid, score)
                muscle_res = self.bio_mem.muscle.find_similar_skills(query_vec, k * 4)
                for vid, score in muscle_res:
                    muscle_hits.setdefault(vid, score)
            else:
                muscle_hits = {}

            hits: List[ReasoningHit] = []
            for vid in candidate_ids:
                h_score = heart_hits.get(vid, 0.0)
                m_score = muscle_hits.get(vid, 0.0)
                g_score = self._graph_score(vid, query_vec)
                a_score = self._ann_score(query_vec, vid)

                # Normalize scores to 0-1
                # Heart and muscle already somewhat normalized
                combined = (
                    self.w_heart * h_score +
                    self.w_muscle * m_score +
                    self.w_graph * g_score +
                    self.w_ann * a_score
                )
                hits.append(ReasoningHit(
                    vid=vid,
                    score=combined,
                    heart_score=h_score,
                    muscle_score=m_score,
                    graph_score=g_score,
                    source="bio+vector"
                ))
            hits.sort(key=lambda x: x.score, reverse=True)
            return hits[:k]

    def augment_context(self, query_vec: np.ndarray, k: int = 5) -> List[str]:
        hits = self.recall(query_vec, k=k)
        return [h.vid for h in hits]

    def update_weights(self, w_heart: float, w_muscle: float, w_graph: float, w_ann: float):
        total = w_heart + w_muscle + w_graph + w_ann
        if total <= 0:
            return
        self.w_heart = w_heart / total
        self.w_muscle = w_muscle / total
        self.w_graph = w_graph / total
        self.w_ann = w_ann / total
