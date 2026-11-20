"""
4-D hypertorus utilities for TRANSLATION_OS_V1

Provides toroidal indexing and neighbour propagation for pattern trees
and weave lattice. The 4-D torus is T⁴ with periodic boundary conditions.
"""

from typing import Tuple, List

class Torus4D:
    def __init__(self, size_per_dim: int = 16):
        self.size = size_per_dim
        self.dim = 4
        self.total = size_per_dim ** 4

    def idx_to_coord(self, idx: int) -> Tuple[int, int, int, int]:
        idx = idx % self.total
        s = self.size
        w = idx // (s**3)
        idx %= s**3
        z = idx // (s**2)
        idx %= s**2
        y = idx // s
        x = idx % s
        return x, y, z, w

    def coord_to_idx(self, coord: Tuple[int, int, int, int]) -> int:
        x, y, z, w = coord
        s = self.size
        return ((w * s + z) * s + y) * s + x

    def neighbours(self, coord: Tuple[int, int, int, int], radius: int = 1) -> List[Tuple[int, int, int, int]]:
        s = self.size
        x, y, z, w = coord
        neigh = []
        # Simple 4-D von Neumann neighbourhood
        for dx in range(-radius, radius+1):
            for dy in range(-radius, radius+1):
                for dz in range(-radius, radius+1):
                    for dw in range(-radius, radius+1):
                        if dx==0 and dy==0 and dz==0 and dw==0:
                            continue
                        nx = (x + dx) % s
                        ny = (y + dy) % s
                        nz = (z + dz) % s
                        nw = (w + dw) % s
                        neigh.append((nx, ny, nz, nw))
        return neigh

    def distance(self, a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
        s = self.size
        dist = 0.0
        for ai, bi in zip(a, b):
            d = abs(ai - bi)
            d = min(d, s - d)
            dist += d*d
        return dist ** 0.5
