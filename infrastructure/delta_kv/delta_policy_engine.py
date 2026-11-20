"""
Delta Policy Engine – decides append/collapse/offload/quantize
"""
from enum import Enum

class CollapseMode(Enum):
    Never = "never"
    Immediate = "immediate"
    ChainLen = "chain_len"
    Adaptive = "adaptive"

class DeltaPolicyEngine:
    def __init__(self, mode=CollapseMode.Adaptive, chain_len_threshold=16):
        self.mode = mode
        self.chain_len_threshold = chain_len_threshold
    
    def decide(self, tile_meta):
        n = len(tile_meta.deltas)
        if self.mode == CollapseMode.Never:
            return "append"
        if self.mode == CollapseMode.Immediate:
            return "collapse"
        if self.mode == CollapseMode.ChainLen:
            return "collapse" if n >= self.chain_len_threshold else "append"
        if self.mode == CollapseMode.Adaptive:
            # simple heuristic: collapse if cold or chain too long
            if not tile_meta.hot and n > 0:
                return "offload"
            if n >= self.chain_len_threshold:
                return "collapse"
            return "append"
        return "append"
