"""
DaxController - Delta shaping brain
Warp-friendly region scheduling, hotness prediction, SIMD magnitude estimation.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import numpy as np
import threading

@dataclass
class DeltaCommand:
    region_id: str
    delta: np.ndarray
    hotness: float
    warp_id: int
    magnitude: float

class DaxController:
    """
    Shapes deltas per region with hotness, warp composition, and overlay magnitude.
    Stable delta shaping for long-running coherence.
    """
    def __init__(self, num_warps: int = 32):
        self.num_warps = num_warps
        self.hotness_counters: Dict[str, int] = {}
        self.lock = threading.RLock()
        self.sat_max = 3  # 2-bit saturating counter range 0-3

    def _update_hotness(self, region_id: str, active: bool):
        with self.lock:
            c = self.hotness_counters.get(region_id, 1)
            if active:
                c = min(self.sat_max, c + 1)
            else:
                c = max(0, c - 1)
            self.hotness_counters[region_id] = c

    def predict_hotness(self, region_id: str) -> float:
        with self.lock:
            c = self.hotness_counters.get(region_id, 1)
            return c / self.sat_max

    def shape_delta(self, region_id: str, base_delta: np.ndarray, warp_id: int, active: bool = True) -> DeltaCommand:
        self._update_hotness(region_id, active)
        hot = self.predict_hotness(region_id)
        mag = float(np.linalg.norm(base_delta)) if base_delta.size else 0.0
        # Speculative delta skipping for cold regions
        if hot < 0.25 and mag < 1e-6:
            shaped = np.zeros_like(base_delta)
        else:
            # SIMD magnitude estimation scaling
            scale = 0.5 + 0.5 * hot
            shaped = base_delta * scale
        return DeltaCommand(
            region_id=region_id,
            delta=shaped,
            hotness=hot,
            warp_id=warp_id % self.num_warps,
            magnitude=mag
        )

    def batch_shape(self, commands: List[tuple]) -> List[DeltaCommand]:
        """
        commands: list of (region_id, base_delta, warp_id, active)
        """
        out = []
        for region_id, base_delta, warp_id, active in commands:
            out.append(self.shape_delta(region_id, base_delta, warp_id, active))
        return out

    def reset_region(self, region_id: str):
        with self.lock:
            self.hotness_counters.pop(region_id, None)
