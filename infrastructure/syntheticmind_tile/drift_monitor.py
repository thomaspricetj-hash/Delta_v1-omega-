"""
DriftMonitor - Semantic drift tracking
SIMD magnitude estimation, rolling window, moving average, threshold detection.
"""
from __future__ import annotations
from collections import deque
import threading
import numpy as np

class DriftMonitor:
    """
    Tracks semantic drift over time to ensure long-run coherence.
    """
    def __init__(self, window_size: int = 1024, threshold: float = 0.15):
        self.window_size = window_size
        self.threshold = threshold
        self.window: deque[float] = deque(maxlen=window_size)
        self.lock = threading.RLock()
        self._sum = 0.0
        self._count = 0

    def _magnitude(self, vec: np.ndarray) -> float:
        # SIMD magnitude estimation
        return float(np.linalg.norm(vec))

    def update(self, vec: np.ndarray, ref_vec: np.ndarray | None = None) -> float:
        mag = self._magnitude(vec)
        if ref_vec is not None:
            diff = self._magnitude(vec - ref_vec)
            score = diff / (mag + 1e-9)
        else:
            # Self drift: deviation from running mean magnitude
            score = abs(mag - self._moving_average()) if self._count else 0.0
        with self.lock:
            self.window.append(score)
            self._sum += score
            self._count += 1
            if len(self.window) > self.window_size:
                # sum maintenance would need removal; simplified
                pass
        return score

    def _moving_average(self) -> float:
        if not self.window:
            return 0.0
        return float(sum(self.window) / len(self.window))

    def drift_score(self) -> float:
        return self._moving_average()

    def is_drifting(self) -> bool:
        return self.drift_score() > self.threshold

    def reset(self):
        with self.lock:
            self.window.clear()
            self._sum = 0.0
            self._count = 0
