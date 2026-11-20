"""
DAX Registry + Hybrid BitDrop Compression
Plug-and-play delta registry for TRANSLATION_OS_V1
"""
import json
from pathlib import Path
from typing import Dict, Any
import numpy as np

class HybridBitDrop:
    """Simple Python shim for V3/V5 Hybrid BitDrop compression.
    In production this would call the Rust engine. Here we use BitDrop v2 as proxy.
    """
    @staticmethod
    def compress(data: bytes) -> bytes:
        try:
            from bitdrop_v2_integration import compress_data
            return compress_data(data)
        except Exception:
            return data

    @staticmethod
    def decompress(data: bytes) -> bytes:
        # Placeholder - BitDrop v2 is not reversible in this shim
        # In real deployment, use Rust engine
        return data

class CompressedDelta:
    def __init__(self, delta_id: str, payload: bytes, is_dax: bool = True):
        self.id = delta_id
        self.is_dax = is_dax
        self.compressed = HybridBitDrop.compress(payload)

    def decompress(self) -> bytes:
        return HybridBitDrop.decompress(self.compressed)

class DAXRegistry:
    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.index_path = self.base_path / "index.json"
        self.deltas: Dict[str, CompressedDelta] = {}
        self._load_index()

    def _load_index(self):
        if self.index_path.exists():
            with open(self.index_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Rehydrate deltas from disk - simplified
                pass

    def _save_index(self):
        with open(self.index_path, 'w', encoding='utf-8') as f:
            json.dump({"deltas": list(self.deltas.keys())}, f)

    def add_delta(self, delta_id: str, delta_bytes: bytes, is_dax: bool = True):
        comp = CompressedDelta(delta_id, delta_bytes, is_dax)
        self.deltas[delta_id] = comp
        # Persist compressed payload with double compression: Hybrid BitDrop -> BitDrop v2
        payload = comp.compressed
        try:
            from bitdrop_v2_integration import compress_data
            payload = compress_data(payload)
        except Exception:
            pass
        out_path = self.base_path / f"{delta_id}.bd"
        out_path.write_bytes(payload)
        self._save_index()

    def reconstruct_state(self, delta_id: str, base_state: bytes) -> bytes:
        if delta_id not in self.deltas:
            raise KeyError(delta_id)
        delta_bytes = self.deltas[delta_id].decompress()
        # XOR reconstruction
        base_arr = bytearray(base_state)
        delta_arr = bytearray(delta_bytes)
        # Pad to same length
        max_len = max(len(base_arr), len(delta_arr))
        base_arr += b'\x00' * (max_len - len(base_arr))
        delta_arr += b'\x00' * (max_len - len(delta_arr))
        out = bytes(b ^ d for b, d in zip(base_arr, delta_arr))
        return out

    def collapse_all(self, base_state: bytes, delta_ids: list) -> bytes:
        state = bytearray(base_state)
        for did in delta_ids:
            delta_bytes = self.deltas[did].decompress()
            # XOR apply
            max_len = max(len(state), len(delta_bytes))
            state += b'\x00' * (max_len - len(state))
            delta_arr = bytearray(delta_bytes) + b'\x00' * (max_len - len(delta_bytes))
            state = bytearray(b ^ d for b, d in zip(state, delta_arr))
        return bytes(state)

    def stats(self) -> Dict[str, Any]:
        total_raw = 0
        total_comp = 0
        for d in self.deltas.values():
            # Approximate
            total_comp += len(d.compressed)
        return {"deltas": len(self.deltas), "compressed_bytes": total_comp}
