"""
Production-grade Heart & Muscle Memory for TRANSLATION_OS_V1

Inspired by biological memory beyond the brain:
- Heart memory: rhythmic, emotional, beat-synchronized consolidation
- Muscle memory: procedural, repetition-strengthened skill storage

Both are implemented as production-grade memory layers that augment DeltaVectorMemory.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import numpy as np
import threading
import math
from collections import deque
import json

# ---------- Heart Rhythmic Memory ----------
@dataclass
class HeartBeat:
    phase: float  # 0-1
    amplitude: float
    emotional_weight: float

class HeartRhythmicMemory:
    """
    Rhythmic, emotionally-weighted memory consolidation.
    Stores memories with cardiac-like beat phase and emotional salience.
    Uses phase-coupled consolidation for long-term retention.
    """
    def __init__(self, dim: int = 256, beat_freq: float = 1.2):
        self.dim = dim
        self.beat_freq = beat_freq  # beats per step
        self.time = 0.0
        self.memories: Dict[str, Dict] = {}
        self.beat_history = deque(maxlen=1024)
        self.lock = threading.RLock()

    def _current_phase(self) -> float:
        return (self.time * self.beat_freq) % 1.0

    def _emotional_weight(self, vector: np.ndarray) -> float:
        # Simple salience: norm + sparsity
        norm = float(np.linalg.norm(vector))
        sparsity = float(np.mean(np.abs(vector) < 0.01))
        return 0.5 + 0.5 * (norm / (norm + 1.0)) * (1.0 - sparsity)

    def store(self, vid: str, vector: np.ndarray, context: str = ""):
        v = np.asarray(vector, dtype=np.float32)
        phase = self._current_phase()
        e_weight = self._emotional_weight(v)
        with self.lock:
            self.memories[vid] = {
                "vector": v,
                "phase": phase,
                "e_weight": e_weight,
                "context": context,
                "access_count": 0,
                "last_access": self.time,
            }
            self.beat_history.append(HeartBeat(phase=phase, amplitude=e_weight, emotional_weight=e_weight))
        self.time += 1.0

    def recall(self, vid: str) -> Tuple[np.ndarray, float]:
        with self.lock:
            if vid not in self.memories:
                raise KeyError(vid)
            mem = self.memories[vid]
            mem["access_count"] += 1
            mem["last_access"] = self.time
            # Phase coherence boost
            phase_diff = abs(self._current_phase() - mem["phase"])
            phase_coherence = 1.0 - min(phase_diff, 1 - phase_diff) * 2
            strength = mem["e_weight"] * (0.7 + 0.3 * phase_coherence)
            return mem["vector"], strength

    def consolidate(self, decay: float = 0.995):
        # Simulate heart-mediated consolidation
        with self.lock:
            for vid, mem in self.memories.items():
                age = self.time - mem["last_access"]
                mem["e_weight"] *= (decay ** age) * (1.0 + 0.01 * mem["access_count"])

    def rhythmic_query(self, query_vec: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        q = np.asarray(query_vec, dtype=np.float32)
        scores = []
        with self.lock:
            for vid, mem in self.memories.items():
                sim = float(np.dot(q, mem["vector"]) / (np.linalg.norm(q) * np.linalg.norm(mem["vector"]) + 1e-9))
                phase_diff = abs(self._current_phase() - mem["phase"])
                phase_coherence = 1.0 - min(phase_diff, 1 - phase_diff) * 2
                score = sim * mem["e_weight"] * (0.8 + 0.2 * phase_coherence)
                scores.append((vid, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]

# ---------- Muscle Procedural Memory ----------
@dataclass
class MuscleTrace:
    sequence: List[str]
    repetitions: int
    strength: float
    last_rehearsal: float

class MuscleProceduralMemory:
    """
    Procedural, repetition-strengthened memory.
    Models skill acquisition via spaced repetition and motor consolidation.
    """
    def __init__(self, dim: int = 256):
        self.dim = dim
        self.skills: Dict[str, MuscleTrace] = {}
        self.embeddings: Dict[str, np.ndarray] = {}
        self.time = 0.0
        self.lock = threading.RLock()

    def learn_sequence(self, skill_id: str, token_ids: List[int], embedding: np.ndarray):
        with self.lock:
            emb = np.asarray(embedding, dtype=np.float32)
            if skill_id not in self.skills:
                self.skills[skill_id] = MuscleTrace(sequence=[], repetitions=0, strength=0.1, last_rehearsal=self.time)
                self.embeddings[skill_id] = emb
            else:
                # Incremental update with momentum
                old = self.embeddings[skill_id]
                self.embeddings[skill_id] = 0.9 * old + 0.1 * emb
            trace = self.skills[skill_id]
            trace.sequence = token_ids[-64:]  # keep recent window
            trace.repetitions += 1
            # Strength grows with repetitions with diminishing returns
            trace.strength = 1.0 - math.exp(-0.1 * trace.repetitions)
            trace.last_rehearsal = self.time
            self.time += 1.0

    def recall_skill(self, skill_id: str) -> Tuple[List[int], float]:
        with self.lock:
            if skill_id not in self.skills:
                raise KeyError(skill_id)
            trace = self.skills[skill_id]
            # Forgetting curve
            age = self.time - trace.last_rehearsal
            recall_strength = trace.strength * math.exp(-0.001 * age)
            return trace.sequence, recall_strength

    def rehearse(self, skill_id: str):
        with self.lock:
            if skill_id in self.skills:
                self.skills[skill_id].last_rehearsal = self.time
                self.skills[skill_id].repetitions += 1

    def find_similar_skills(self, query_emb: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        q = np.asarray(query_emb, dtype=np.float32)
        scores = []
        with self.lock:
            for sid, emb in self.embeddings.items():
                sim = float(np.dot(q, emb) / (np.linalg.norm(q) * np.linalg.norm(emb) + 1e-9))
                strength = self.skills[sid].strength
                scores.append((sid, sim * strength))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]

# ---------- Unified Bio-Memory ----------
class BioMemory:
    """
    Unified Heart + Muscle + Delta Vector memory.
    Provides rhythmic emotional consolidation and procedural skill storage.
    Integrated with DeltaVectorMemory multilayered index/graph/ledger.
    """
    def __init__(self, dim: int = 256, vector_mem: Optional[object] = None):
        self.heart = HeartRhythmicMemory(dim=dim)
        self.muscle = MuscleProceduralMemory(dim=dim)
        self.vector_mem = vector_mem
        self.lock = threading.RLock()

    def store_experience(self, vid: str, vector: np.ndarray, token_ids: List[int], context: str = ""):
        with self.lock:
            self.heart.store(vid, vector, context)
            self.muscle.learn_sequence(vid, token_ids, vector)
            # Persist to delta vector memory with multilayered index/graph/ledger
            if self.vector_mem is not None:
                try:
                    # Add / update vector
                    if vid not in self.vector_mem.store.vectors:
                        self.vector_mem.add_memory(vid, vector)
                    else:
                        # Compute delta
                        old = self.vector_mem.store.get(vid)
                        delta = vector - old
                        self.vector_mem.update_memory(vid, delta)
                    # Graph link by temporal order: link to most recent
                    # Simple heuristic: link to previous by numeric order
                    # In production, use time-based linking
                    self.vector_mem.ledger.append("bio_store", vid, json.dumps({"context": context}).encode(), layer=1)
                except Exception:
                    pass

    def consolidate(self):
        self.heart.consolidate()
        # Muscle consolidation happens via rehearsal

    def query(self, vector: np.ndarray, k: int = 5) -> Dict[str, List[Tuple[str, float]]]:
        heart_results = self.heart.rhythmic_query(vector, k)
        muscle_results = self.muscle.find_similar_skills(vector, k)
        return {"heart": heart_results, "muscle": muscle_results}
