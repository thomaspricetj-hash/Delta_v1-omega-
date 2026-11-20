"""
KvCache – high level API tying allocator + policy + GPU backend
"""
import numpy as np
try:
    from .delta_allocator import DeltaAllocator
    from .delta_policy_engine import DeltaPolicyEngine, CollapseMode
    from .gpu_backend import GPUBackend
except Exception:
    from delta_allocator import DeltaAllocator
    from delta_policy_engine import DeltaPolicyEngine, CollapseMode
    from gpu_backend import GPUBackend

class KvCache:
    def __init__(self, collapse_mode=CollapseMode.Adaptive):
        self.allocator = DeltaAllocator()
        self.policy = DeltaPolicyEngine(mode=collapse_mode)
        self.gpu = GPUBackend()
    
    def alloc_kv_tile(self, shape):
        return self.allocator.alloc_tile(shape)
    
    def apply_kv_update(self, tile_id, delta_payload, delta_type='dense'):
        # delta_payload can be bytes or np array
        tile = self.allocator.tiles.get(tile_id)
        if not tile:
            raise ValueError("tile not found")
        decision = self.policy.decide(tile)
        if decision == "collapse":
            self.allocator.collapse_tile(tile_id)
        # apply delta
        if isinstance(delta_payload, bytes):
            # interpret as float16 bytes
            arr = np.frombuffer(delta_payload, dtype=np.float16)
            # reshape to tile shape if possible
            try:
                arr = arr.reshape(tile.shape)
            except:
                pass
        else:
            arr = np.array(delta_payload)
        self.allocator.apply_delta_raw(tile_id, arr, delta_type)
        # upload to GPU
        self.gpu.upload_delta(tile_id, arr)
        return decision
    
    def materialize_kv(self, tile_id):
        tile = self.allocator.tiles.get(tile_id)
        if not tile:
            return None
        # reconstruct base + deltas
        if tile.deltas:
            # simple sum
            reconstructed = tile.base.astype(np.float32) + np.sum(tile.deltas, axis=0).astype(np.float32)
            reconstructed = reconstructed.astype(np.float16)
        else:
            reconstructed = tile.base
        # upload to GPU for attention
        self.gpu.materialize(tile_id, reconstructed)
        return reconstructed
    
    def get_compression_ratio(self, tile_id):
        tile = self.allocator.tiles.get(tile_id)
        if not tile:
            return 0.0
        base_bytes = tile.base.nbytes
        delta_bytes = sum(d.nbytes for d in tile.deltas)
        naive = base_bytes * (1 + len(tile.deltas))
        delta_backed = base_bytes + delta_bytes
        return naive / max(1, delta_backed)
