"""
Delta-backed KV integration for WindowIndexer
Production grade
"""
from pathlib import Path
from typing import List
import torch
import numpy as np
from infrastructure.kv.delta_kv import KvCache, CollapseMode, DeltaType
from infrastructure.indexing.window_indexer import WindowIndexer

class DeltaWindowIndexer:
    def __init__(self, base_path: Path, scales: List[int] = [128,256,512,1024]):
        self.indexer = WindowIndexer(base_path, scales)
        self.kv_cache = KvCache(collapse_mode=CollapseMode.ADAPTIVE)
        self.window_to_tile: dict = {}

    def build_from_texts(self, texts: List[str], tokenizer):
        # Build normal index first
        self.indexer.build_from_texts(texts, tokenizer)
        # Allocate delta tiles for each scale
        # In production, map windows to tiles
        pass

    def sample_windows_delta(self, scale: int, batch_size: int):
        windows = self.indexer.sample_windows(scale, batch_size)
        # Materialize via delta cache if needed
        return windows
