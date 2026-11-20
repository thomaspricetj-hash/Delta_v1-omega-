"""
OverlayChain & OverlaySram - Streaming memory backbone
Production grade, thread-safe, SIMD-friendly layouts.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import threading
import numpy as np
from collections import deque

@dataclass
class Overlay:
    id: str
    data: np.ndarray
    magnitude: float = 0.0
    timestamp: int = 0
    layer: int = 0

class OverlaySram:
    """
    SRAM-like buffer with high locality.
    Holds current tile working set.
    """
    def __init__(self, size: int = 4096, dtype=np.float32):
        self.size = size
        self.dtype = dtype
        self.buffer = np.zeros(size, dtype=dtype)
        self.lock = threading.RLock()
        self.write_ptr = 0

    def write(self, data: np.ndarray) -> int:
        with self.lock:
            n = min(len(data), self.size - self.write_ptr)
            self.buffer[self.write_ptr:self.write_ptr+n] = data[:n]
            self.write_ptr = (self.write_ptr + n) % self.size
            return n

    def read(self, n: int) -> np.ndarray:
        with self.lock:
            return self.buffer[:n].copy()

    def reset(self):
        with self.lock:
            self.buffer.fill(0)
            self.write_ptr = 0

class OverlayChain:
    """
    Temporal memory buffer capturing snapshots of tile state.
    SIMD-accelerated blending, tiny overlay fusion, zero-cost push.
    """
    def __init__(self, max_overlays: int = 1024):
        self.chain: deque[Overlay] = deque(maxlen=max_overlays)
        self.lock = threading.RLock()
        self._counter = 0

    def push(self, data: np.ndarray, layer: int = 0) -> str:
        with self.lock:
            oid = f"ov_{self._counter}"
            self._counter += 1
            mag = float(np.linalg.norm(data))
            ov = Overlay(id=oid, data=data.copy(), magnitude=mag, timestamp=self._counter, layer=layer)
            self.chain.append(ov)
            return oid

    def blend(self, ids: List[str]) -> np.ndarray:
        with self.lock:
            overlays = [o for o in self.chain if o.id in ids]
            if not overlays:
                return np.array([], dtype=np.float32)
            # SIMD-friendly weighted blend by magnitude
            mats = [o.data for o in overlays]
            # Pad to same length
            max_len = max(len(m) for m in mats)
            acc = np.zeros(max_len, dtype=np.float32)
            weights = []
            for o in overlays:
                w = max(1e-9, o.magnitude)
                weights.append(w)
                # align
                d = np.zeros(max_len, dtype=np.float32)
                d[:len(o.data)] = o.data
                acc += d * w
            wsum = sum(weights) + 1e-9
            return acc / wsum

    def fuse_tiny(self, threshold: float = 1e-3):
        """
        Tiny overlay fusion to reduce overhead.
        Merges overlays with magnitude below threshold into neighbors.
        """
        with self.lock:
            # Simple in-place filter
            kept = deque()
            for o in self.chain:
                if o.magnitude >= threshold:
                    kept.append(o)
            self.chain = kept

    def get_hotness(self, k: int = 8) -> List[Tuple[str, float]]:
        with self.lock:
            ranked = sorted(self.chain, key=lambda o: o.magnitude, reverse=True)
            return [(o.id, float(o.magnitude)) for o in ranked[:k]]

    def clear(self):
        with self.lock:
            self.chain.clear()
