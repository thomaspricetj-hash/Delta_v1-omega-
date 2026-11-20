"""
CollapseUnit - SRAM -> OverlayChain conversion
AVX2 normalization, tiny overlay fusion, streaming-friendly.
"""
from __future__ import annotations
import numpy as np
from .overlay_chain import OverlayChain, OverlaySram

class CollapseUnit:
    """
    Converts SRAM working set into OverlayChain with normalization.
    Stable magnitudes improve drift detection and hotness prediction.
    """
    def __init__(self, chain: OverlayChain, norm_type: str = "l1"):
        self.chain = chain
        self.norm_type = norm_type

    def collapse(self, sram: OverlaySram, layer: int = 0) -> str:
        data = sram.read(sram.size)
        # Remove zeros to avoid empty overlays
        nz = data[np.abs(data) > 1e-12]
        if nz.size == 0:
            nz = data
        # AVX2-style L1 norm scaling
        if self.norm_type == "l1":
            norm = float(np.sum(np.abs(nz))) + 1e-9
            scaled = nz / norm
        elif self.norm_type == "l2":
            norm = float(np.linalg.norm(nz)) + 1e-9
            scaled = nz / norm
        else:
            scaled = nz
        oid = self.chain.push(scaled, layer=layer)
        # Tiny fusion maintenance
        self.chain.fuse_tiny(threshold=1e-4)
        return oid

    def collapse_batch(self, srams: List[OverlaySram], layer: int = 0) -> List[str]:
        oids = []
        for s in srams:
            oids.append(self.collapse(s, layer))
        return oids
