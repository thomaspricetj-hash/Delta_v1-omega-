"""
Production-grade Multilayered Grid Heatmap for TRANSLATION_OS_V1
Spatial activation maps with diffusion, decay, and hot-region gating.
Used for sparse routing, memory candidate filtering, and MDWE heat gating.
"""
from __future__ import annotations
import threading
import numpy as np
from typing import Tuple, List, Dict, Optional
import hashlib

class GridHeatmap:
    """
    2D heatmap with exponential decay and local diffusion.
    Thread-safe, CPU-friendly, suitable for per-layer activation tracking.
    """
    def __init__(self, width: int = 64, height: int = 64, decay: float = 0.995, spread: float = 0.12):
        self.w = width
        self.h = height
        self.decay = float(decay)
        self.spread = float(spread)
        self.heat = np.zeros((height, width), dtype=np.float32)
        self.lock = threading.RLock()
        # Per-cell last update for age-based weighting
        self.age = np.zeros((height, width), dtype=np.float32)

    def _coord_from_id(self, vid: str) -> Tuple[int, int]:
        # Deterministic hashing to grid cell
        h = hashlib.sha256(vid.encode()).digest()
        x = int.from_bytes(h[:4], 'big') % self.w
        y = int.from_bytes(h[4:8], 'big') % self.h
        return x, y

    def decay_step(self):
        with self.lock:
            self.heat *= self.decay
            self.age += 1.0

    def add_activation(self, vid: str, value: float, radius: int = 1):
        x, y = self._coord_from_id(vid)
        with self.lock:
            # Central cell
            self.heat[y, x] += float(value)
            self.age[y, x] = 0.0
            # Diffusion to neighbours with Gaussian bell curve
            for dy in range(-radius, radius+1):
                for dx in range(-radius, radius+1):
                    nx = x + dx
                    ny = y + dy
                    if 0 <= nx < self.w and 0 <= ny < self.h:
                        dist2 = dx*dx + dy*dy
                        # Gaussian bell curve with sigma = radius
                        w = self.spread * np.exp(-dist2 / (2.0 * max(1, radius*radius)))
                        self.heat[ny, nx] += float(value) * w
                        self.age[ny, nx] = 0.0

    def get_heat(self, vid: str) -> float:
        x, y = self._coord_from_id(vid)
        with self.lock:
            return float(self.heat[y, x])

    def hot_regions(self, k: int = 16, threshold: float = 1e-3) -> List[Tuple[int, int, float]]:
        with self.lock:
            flat = self.heat.flatten()
            if flat.size == 0:
                return []
            # Use argpartition for O(n)
            if k >= flat.size:
                idx = np.where(flat > threshold)[0]
            else:
                idx = np.argpartition(flat, -k)[-k:]
            idx = idx[flat[idx] > threshold]
            regions = []
            for i in idx:
                y, x = divmod(i, self.w)
                regions.append((x, y, float(self.heat[y, x])))
            regions.sort(key=lambda t: t[2], reverse=True)
            return regions[:k]

    def get_cells_in_radius(self, vid: str, radius: int = 2) -> List[str]:
        # Returns list of vids? Not possible without reverse map.
        # Return grid coords instead.
        x, y = self._coord_from_id(vid)
        cells = []
        for dy in range(-radius, radius+1):
            for dx in range(-radius, radius+1):
                nx = x + dx
                ny = y + dy
                if 0 <= nx < self.w and 0 <= ny < self.h:
                    cells.append((nx, ny))
        return cells

    def reset(self):
        with self.lock:
            self.heat.fill(0.0)
            self.age.fill(0.0)

class MultilayeredHeatmap:
    """
    Stack of heatmaps per layer / modality.
    Provides unified hot-region gating across layers.
    """
    def __init__(self, layers: int = 4, width: int = 64, height: int = 64):
        self.maps: Dict[int, GridHeatmap] = {
            i: GridHeatmap(width, height) for i in range(layers)
        }
        self.layer_weights = np.ones(layers, dtype=np.float32)

    def add_activation(self, layer: int, vid: str, value: float, radius: int = 1):
        if layer in self.maps:
            self.maps[layer].add_activation(vid, value, radius)

    def decay_all(self):
        for m in self.maps.values():
            m.decay_step()

    def hot_regions(self, layer: Optional[int] = None, k: int = 16, threshold: float = 1e-3) -> List[Tuple[int, int, float, int]]:
        results = []
        layers = [layer] if layer is not None else self.maps.keys()
        for l in layers:
            regions = self.maps[l].hot_regions(k, threshold)
            for x, y, v in regions:
                results.append((x, y, float(v), l))
        results.sort(key=lambda t: t[2], reverse=True)
        return results[:k]

    def filter_candidates_by_heat(self, candidates: List[str], layer: int = 0, top_p: float = 0.3) -> List[str]:
        if not candidates:
            return []
        hm = self.maps.get(layer)
        if hm is None:
            return candidates
        heats = [(vid, hm.get_heat(vid)) for vid in candidates]
        heats.sort(key=lambda t: t[1], reverse=True)
        keep_n = max(1, int(len(candidates) * top_p))
        return [vid for vid, _ in heats[:keep_n]]

    def get_heat(self, layer: int, vid: str) -> float:
        return self.maps.get(layer, GridHeatmap()).get_heat(vid)
