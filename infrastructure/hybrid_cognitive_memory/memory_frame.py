"""
MemoryFrame - low-level neural-style substrate
Slices, grids, confidence, decay, conflict resolution
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import math
import time

@dataclass
class Cell:
    value: float = 0.0
    confidence: float = 0.0
    timestamp: float = field(default_factory=time.time)
    decay_rate: float = 0.001

    def decay(self):
        age = time.time() - self.timestamp
        self.confidence *= math.exp(-self.decay_rate * age)
        self.value *= self.confidence

@dataclass
class Slice:
    name: str
    grid: Dict[Tuple[int,int], Cell] = field(default_factory=dict)

    def get(self, x, y):
        return self.grid.get((x,y), Cell())

    def set(self, x, y, value, confidence):
        cell = self.grid.get((x,y), Cell())
        cell.value = value
        cell.confidence = confidence
        cell.timestamp = time.time()
        self.grid[(x,y)] = cell

class MemoryFrame:
    def __init__(self, policy=None):
        self.slices = {
            "semantic": Slice("semantic"),
            "relational": Slice("relational"),
            "temporal": Slice("temporal"),
            "spatial": Slice("spatial"),
        }
        self.policy = policy or {"decay": 0.001, "conflict_threshold": 0.3}

    def write(self, slice_name, x, y, value, confidence):
        if slice_name in self.slices:
            self.slices[slice_name].set(x, y, value, confidence)

    def read(self, slice_name, x, y):
        if slice_name in self.slices:
            cell = self.slices[slice_name].get(x, y)
            cell.decay()
            return cell
        return Cell()

    def decay_all(self):
        for s in self.slices.values():
            for cell in s.grid.values():
                cell.decay()

    def resolve_conflict(self, cells: List[Cell]):
        # Weighted average by confidence
        total_conf = sum(c.confidence for c in cells)
        if total_conf == 0:
            return 0.0
        weighted = sum(c.value * c.confidence for c in cells)
        return weighted / total_conf
