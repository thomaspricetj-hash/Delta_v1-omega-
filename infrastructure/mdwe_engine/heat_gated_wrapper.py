"""
Heat-gated MDWE projector
Integrates MultilayeredHeatmap for spatially-aware sparse activation gating.
Production-grade, thread-safe, CPU-friendly.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np
from typing import Optional

try:
    from .torch_wrapper import MDWESparseProjector
    from ..heatmap.grid_heatmap import MultilayeredHeatmap
    MDWE_AVAILABLE = True
except Exception:
    MDWE_AVAILABLE = False
    MDWESparseProjector = None
    MultilayeredHeatmap = None

class HeatGatedMDWEProjector(nn.Module):
    """
    Wraps MDWESparseProjector with heatmap-based gating.
    Hot grid cells bias neuron activation, cold cells are suppressed.
    """
    def __init__(self, num_neurons: int = 1024, dim: int = 256, heatmap: Optional[MultilayeredHeatmap] = None, gate_strength: float = 0.3):
        super().__init__()
        if not MDWE_AVAILABLE:
            raise RuntimeError("MDWE not available")
        self.mdwe = MDWESparseProjector(num_neurons=num_neurons, dim=dim)
        self.heatmap = heatmap
        self.gate_strength = float(gate_strength)
        # Neuron to grid mapping for heat gating
        self.num_neurons = num_neurons
        # Simple deterministic mapping: neuron idx -> grid coord via hash
        import hashlib
        self.neuron_coords = []
        for i in range(num_neurons):
            h = hashlib.sha256(str(i).encode()).digest()
            x = int.from_bytes(h[:4], 'big') % 64
            y = int.from_bytes(h[4:8], 'big') % 64
            self.neuron_coords.append((x, y))

    def forward(self, x: torch.Tensor):
        # Base MDWE projection
        out = self.mdwe(x)
        if self.heatmap is None:
            return out
        # Apply heat gating per sample
        with torch.no_grad():
            # Get hot regions from layer 0 heatmap
            hot = self.heatmap.maps[0].hot_regions(k=32, threshold=1e-3)
            hot_set = set((x, y) for x, y, _ in hot)
            # Build gating mask for neurons
            mask = torch.ones(self.num_neurons, dtype=torch.float32, device=out.device)
            for i, (gx, gy) in enumerate(self.neuron_coords):
                if (gx, gy) not in hot_set:
                    # Suppress cold neurons
                    mask[i] = 1.0 - self.gate_strength
            # Reshape mask to broadcast over batch and seq
            # MDWE output shape is [B, S, D]; we assume D <= num_neurons for simplicity
            # Apply gating via elementwise multiplication on the feature dim
            # If D < num_neurons, we slice mask
            d = out.shape[-1]
            if d <= self.num_neurons:
                mask = mask[:d]
            else:
                # Pad mask
                pad = torch.ones(d - self.num_neurons, dtype=torch.float32, device=out.device)
                mask = torch.cat([mask, pad])
            mask = mask.view(1, 1, -1)
            out = out * mask
        return out

    def load_snapshot(self, path):
        self.mdwe.load_snapshot(path)

    def set_awareness_gates(self, gates):
        # gates: [B,3] confidence, novelty, stability
        # Store for next forward to modulate gate strength
        self._awareness_gates = gates.detach().cpu()

    def save_snapshot(self, path):
        self.mdwe.save_snapshot(path)
