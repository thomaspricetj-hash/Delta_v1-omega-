"""
GPU Backend stub – portable across vendors
"""
class GPUBackend:
    def __init__(self):
        self.buffers = {}
    
    def upload_delta(self, tile_id, delta):
        # stub: store reference
        self.buffers.setdefault(tile_id, {})['deltas'] = delta
    
    def materialize(self, tile_id, tensor):
        self.buffers.setdefault(tile_id, {})['materialized'] = tensor
        # In real impl: copy to GPU, run compute shader
        return True
    
    def collapse_on_gpu(self, tile_id):
        # future: WGSL kernels
        return True
