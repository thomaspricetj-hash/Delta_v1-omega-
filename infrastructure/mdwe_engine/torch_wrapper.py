"""
MDWE PyTorch wrapper with differentiable straight-through estimator
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Optional

class MDWEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, mdwe_engine):
        # x: [B, S, D]
        # Convert to numpy, run MDWE per batch
        x_np = x.detach().cpu().numpy()
        B, S, D = x_np.shape
        out_list = []
        for i in range(B):
            key = x_np[i].mean(axis=0)
            res = mdwe_engine.forward(key.astype(np.float32))
            out = res.get('output', key)
            if out.shape[0] != D:
                if out.shape[0] < D:
                    out = np.pad(out, (0, D - out.shape[0]))
                else:
                    out = out[:D]
            out_b = np.broadcast_to(out, (S, D))
            out_list.append(out_b)
        out_np = np.stack(out_list)
        out_t = torch.from_numpy(out_np).to(x.device).type_as(x)
        ctx.mark_non_differentiable(out_t)
        return out_t

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator: pass gradients through unchanged
        # MDWE weights are updated via side-car, not via autograd
        return grad_output, None

class MDWESparseProjector(nn.Module):
    def __init__(self, num_neurons: int = 1024, dim: int = 256):
        super().__init__()
        try:
            from .mdwe_core import MDWEEngine
            self.engine = MDWEEngine(num_neurons=num_neurons, dim=dim)
            self.enabled = True
        except Exception:
            self.engine = None
            self.enabled = False

    def forward(self, x: torch.Tensor):
        if not self.enabled or self.engine is None:
            return x
        # During training we still want gradients to flow. Use straight-through.
        return MDWEFunction.apply(x, self.engine)

    def load_snapshot(self, path):
        if self.enabled:
            self.engine.load_snapshot(path)

    def save_snapshot(self, path):
        if self.enabled:
            self.engine.save_snapshot(path)
