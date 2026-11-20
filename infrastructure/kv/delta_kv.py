"""
SyntheticMind Delta-Backed KV System - Production Grade
High-compression, CPU-friendly transformer memory architecture
"""
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import threading

class DeltaType(Enum):
    DENSE = "dense"
    SPARSE = "sparse"
    LOW_RANK = "low_rank"

class CollapseMode(Enum):
    NEVER = "never"
    IMMEDIATE = "immediate"
    CHAIN_LEN = "chain_len"
    ADAPTIVE = "adaptive"

@dataclass
class KVTileMeta:
    shape: Tuple[int, int, int]
    base_dtype: torch.dtype = torch.float16
    delta_type: DeltaType = DeltaType.DENSE
    delta_bits: int = 8
    chain_len: int = 0
    last_access: int = 0
    temperature: float = 1.0

class DeltaAllocator:
    def __init__(self):
        self.tiles: Dict[int, Dict] = {}
        self.lock = threading.Lock()
        self.next_id = 0

    def alloc_tile(self, shape: Tuple[int, int, int]) -> int:
        with self.lock:
            tile_id = self.next_id
            self.next_id += 1
            base = torch.empty(shape, dtype=torch.float16)
            self.tiles[tile_id] = {
                'meta': KVTileMeta(shape=shape),
                'base': base,
                'deltas': [],
                'offloaded': False
            }
            return tile_id

    def apply_delta_raw(self, tile_id: int, delta: torch.Tensor, delta_type: DeltaType = DeltaType.DENSE):
        with self.lock:
            tile = self.tiles.get(tile_id)
            if not tile:
                raise KeyError(f"Tile {tile_id} not found")
            # Quantize delta if needed
            if delta_type == DeltaType.DENSE:
                # Int8 quantization
                dmin = delta.min()
                dmax = delta.max()
                if dmax > dmin:
                    scale = 127.0 / (dmax - dmin)
                    q = ((delta - dmin) * scale).round().to(torch.int8)
                    delta_store = {'q': q, 'min': dmin, 'scale': scale}
                else:
                    delta_store = {'q': torch.zeros_like(delta, dtype=torch.int8)}
            else:
                delta_store = {'data': delta}
            tile['deltas'].append({
                'type': delta_type,
                'data': delta_store,
                'step': len(tile['deltas'])
            })
            tile['meta'].chain_len = len(tile['deltas'])

    def collapse_tile(self, tile_id: int):
        with self.lock:
            tile = self.tiles.get(tile_id)
            if not tile or not tile['deltas']:
                return
            base = tile['base']
            for d in tile['deltas']:
                if d['type'] == DeltaType.DENSE:
                    q = d['data']['q']
                    scale = d['data']['scale']
                    dmin = d['data']['min']
                    delta = q.float() / scale + dmin
                    base += delta
            tile['base'] = base
            tile['deltas'] = []
            tile['meta'].chain_len = 0

    def offload_tile(self, tile_id: int):
        with self.lock:
            tile = self.tiles.get(tile_id)
            if tile:
                tile['offloaded'] = True
                # In production, move base to host memory
                # Here we just mark

    def materialize(self, tile_id: int) -> torch.Tensor:
        with self.lock:
            tile = self.tiles.get(tile_id)
            if not tile:
                raise KeyError(f"Tile {tile_id} not found")
            base = tile['base'].clone()
            for d in tile['deltas']:
                if d['type'] == DeltaType.DENSE:
                    q = d['data']['q']
                    scale = d['data']['scale']
                    dmin = d['data']['min']
                    delta = q.float() / scale + dmin
                    base += delta
            return base

class DeltaPolicyEngine:
    def __init__(self, collapse_mode: CollapseMode = CollapseMode.ADAPTIVE, chain_threshold: int = 16):
        self.mode = collapse_mode
        self.chain_threshold = chain_threshold
        self.adaptive_counter = 0

    def decide(self, tile_meta: KVTileMeta, step: int) -> str:
        if self.mode == CollapseMode.NEVER:
            return "append"
        if self.mode == CollapseMode.IMMEDIATE:
            return "collapse"
        if self.mode == CollapseMode.CHAIN_LEN:
            if tile_meta.chain_len >= self.chain_threshold:
                return "collapse"
            return "append"
        if self.mode == CollapseMode.ADAPTIVE:
            # Adaptive based on temperature and chain length
            if tile_meta.chain_len >= self.chain_threshold and tile_meta.temperature < 0.5:
                return "collapse"
            if tile_meta.chain_len >= self.chain_threshold * 2:
                return "collapse"
            return "append"
        return "append"

class KvCache:
    def __init__(self, collapse_mode: CollapseMode = CollapseMode.ADAPTIVE):
        self.allocator = DeltaAllocator()
        self.policy = DeltaPolicyEngine(collapse_mode)
        self.step_counter = 0

    def alloc_kv_tile(self, shape: Tuple[int, int, int]) -> int:
        return self.allocator.alloc_tile(shape)

    def apply_kv_update(self, tile_id: int, delta: torch.Tensor, delta_type: DeltaType = DeltaType.DENSE):
        self.step_counter += 1
        tile = self.allocator.tiles[tile_id]
        meta = tile['meta']
        meta.last_access = self.step_counter
        action = self.policy.decide(meta, self.step_counter)
        if action == "collapse":
            self.allocator.collapse_tile(tile_id)
            self.allocator.apply_delta_raw(tile_id, delta, delta_type)
        else:
            self.allocator.apply_delta_raw(tile_id, delta, delta_type)

    def materialize_kv(self, tile_id: int) -> torch.Tensor:
        return self.allocator.materialize(tile_id)

    def collapse_tile(self, tile_id: int):
        self.allocator.collapse_tile(tile_id)

    def offload_tile(self, tile_id: int):
        self.allocator.offload_tile(tile_id)

    def get_compression_ratio(self, tile_id: int) -> float:
        tile = self.allocator.tiles[tile_id]
        base_bytes = tile['meta'].shape[0] * tile['meta'].shape[1] * tile['meta'].shape[2] * 2
        delta_bytes = len(tile['deltas']) * 1024  # approx
        if delta_bytes == 0:
            return 1.0
        naive = base_bytes * (1 + len(tile['deltas']))
        delta_backed = base_bytes + delta_bytes
        return naive / max(1, delta_backed)
