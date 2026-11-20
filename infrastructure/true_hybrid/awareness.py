"""
Production-grade Self-Awareness Head for TRANSLATION_OS_V1
Introspective meta-head that senses internal state and outputs gating signals.
Architectural awareness, not training-only monitoring.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from typing import Tuple

class AwarenessHead(nn.Module):
    """
    Self-awareness module that maps hidden states to meta-gates:
    confidence, novelty, stability
    Gates are used to modulate MDWE, predictor, and memory recall in-model.
    """
    def __init__(self, dim: int = 256, hidden: int = 128):
        super().__init__()
        self.dim = dim
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 3)
        )
        # ReLU + LayerNorm bottleneck for gating stability and sparsity
        self.post_norm = nn.LayerNorm(3)
        self.relu = nn.ReLU(inplace=True)
        # Learnable temperature for gating sharpness
        self.temp = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B, S, D]
        Returns:
          gates: [B, 3] in [0,1] for confidence, novelty, stability
          meta_vec: [B, 3] raw logits
        """
        # Mean pool over sequence
        h = self.norm(x)
        h_pooled = self.pool(h.transpose(1, 2)).squeeze(-1)  # [B, D]
        meta_logits = self.net(h_pooled)  # [B, 3]
        # ReLU + LayerNorm bottleneck for stability/sparsity before gating
        meta_logits = self.post_norm(meta_logits)
        meta_logits = self.relu(meta_logits)
        gates = torch.sigmoid(meta_logits / self.temp.clamp(min=0.1))
        return gates, meta_logits

    def extra_repr(self):
        return f"dim={self.dim}"
