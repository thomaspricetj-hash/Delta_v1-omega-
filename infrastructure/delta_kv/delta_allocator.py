"""
Delta Allocator – CPU management of base tiles and delta chains
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List

@dataclass
class TileMeta:
    tile_id: int
    shape: tuple
    base: np.ndarray = None
    deltas: List[np.ndarray] = field(default_factory=list)
    delta_types: List[str] = field(default_factory=list)
    hot: bool = True

class DeltaAllocator:
    def __init__(self):
        self.tiles: Dict[int, TileMeta] = {}
        self.next_id = 0
    
    def alloc_tile(self, shape):
        base = np.zeros(shape, dtype=np.float16)
        meta = TileMeta(tile_id=self.next_id, shape=shape, base=base)
        self.tiles[self.next_id] = meta
        self.next_id += 1
        return meta.tile_id
    
    def apply_delta_raw(self, tile_id, delta, delta_type='dense'):
        tile = self.tiles.get(tile_id)
        if not tile:
            raise ValueError("tile not found")
        tile.deltas.append(np.array(delta, dtype=np.float16))
        tile.delta_types.append(delta_type)
    
    def collapse_tile(self, tile_id):
        tile = self.tiles.get(tile_id)
        if not tile:
            return
        # simple accumulation
        if tile.deltas:
            acc = np.sum(tile.deltas, axis=0)
            tile.base = (tile.base.astype(np.float32) + acc.astype(np.float32)).astype(np.float16)
            tile.deltas.clear()
            tile.delta_types.clear()
    
    def offload_tile(self, tile_id):
        tile = self.tiles.get(tile_id)
        if tile:
            tile.hot = False
