"""
Lazy dequantizing Dual-Weight Linear using Layered Codebook
Weights stay compressed in RAM as indices + small layered codebooks.
Dequantization happens on forward and is cached until weights change.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional
from infrastructure.true_hybrid.layered_codebook_dual import LayeredCodebook
from infrastructure.true_hybrid.tree_store import TreeCodebook

class CodebookDualWeightLinear(nn.Module):
    """
    DualWeightLinear with optional codebook compression and lazy dequantization.
    When use_codebook=True, master/delta weights are stored as indices into LayeredCodebook.
    Forward dequantizes on demand and caches the result.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        delta_decay: float = 0.01,
        delta_clip: float = 1.0,
        use_codebook: bool = False,
        codebook_capacity: int = 1024,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.delta_decay = delta_decay
        self.delta_clip = delta_clip
        self.use_codebook = use_codebook
        self.codebook_capacity = codebook_capacity

        # Codebooks - use tree-structured codebook for automatic node splitting
        self.master_codebook = TreeCodebook(capacity_per_node=codebook_capacity, dir_path=f"cb_master_{id(self)}")
        self.delta_codebook = TreeCodebook(capacity_per_node=codebook_capacity, dir_path=f"cb_delta_{id(self)}")

        if use_codebook:
            # Indices buffers, shape = weight shape
            shape = (out_features, in_features)
            self.register_buffer("master_layer_ids", torch.zeros(shape, dtype=torch.int32))
            self.register_buffer("master_code_ids", torch.zeros(shape, dtype=torch.int32))
            self.register_buffer("delta_layer_ids", torch.zeros(shape, dtype=torch.int32))
            self.register_buffer("delta_code_ids", torch.zeros(shape, dtype=torch.int32))
            # Bias as normal parameters for simplicity
            if bias:
                self.master_bias = nn.Parameter(torch.empty(out_features))
                self.delta_bias = nn.Parameter(torch.zeros(out_features))
            else:
                self.register_parameter('master_bias', None)
                self.register_parameter('delta_bias', None)
            self._master_weight_cache = None
            self._delta_weight_cache = None
            self._cache_valid = False
            nn.init.uniform_(self.master_bias, -0.1, 0.1) if bias else None
        else:
            # Classic dual weight parameters
            self.master_weight = nn.Parameter(torch.empty(out_features, in_features))
            self.delta_weight = nn.Parameter(torch.zeros(out_features, in_features))
            if bias:
                self.master_bias = nn.Parameter(torch.empty(out_features))
                self.delta_bias = nn.Parameter(torch.zeros(out_features))
            else:
                self.register_parameter('master_bias', None)
                self.register_parameter('delta_bias', None)
            self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.master_weight, a=math.sqrt(5))
        if self.master_bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.master_weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.master_bias, -bound, bound)

    def convert_from_weights(self, master_weight: torch.Tensor, delta_weight: torch.Tensor):
        """
        Compress existing full-precision weights into layered codebooks and indices.
        Call once after training or loading.
        """
        if not self.use_codebook:
            raise RuntimeError("convert_from_weights only valid with use_codebook=True")
        master_np = master_weight.detach().cpu().numpy().astype('float32')
        delta_np = delta_weight.detach().cpu().numpy().astype('float32')
        # Build codebooks
        self.master_codebook.fit_from_data(master_np)
        self.delta_codebook.fit_from_data(delta_np)
        # Quantize
        m_layer_ids, m_code_ids = self.master_codebook.quantize(master_np)
        d_layer_ids, d_code_ids = self.delta_codebook.quantize(delta_np)
        # Store as buffers
        self.master_layer_ids.copy_(torch.from_numpy(m_layer_ids.reshape(self.master_layer_ids.shape)).int())
        self.master_code_ids.copy_(torch.from_numpy(m_code_ids.reshape(self.master_code_ids.shape)).int())
        self.delta_layer_ids.copy_(torch.from_numpy(d_layer_ids.reshape(self.delta_layer_ids.shape)).int())
        self.delta_code_ids.copy_(torch.from_numpy(d_code_ids.reshape(self.delta_code_ids.shape)).int())
        self._cache_valid = False

    def _dequantize_weight(self, layer_ids_buf, code_ids_buf, codebook):
        # Convert buffers to numpy
        layer_ids = layer_ids_buf.detach().cpu().numpy().flatten()
        code_ids = code_ids_buf.detach().cpu().numpy().flatten()
        deq = codebook.dequantize(layer_ids, code_ids)
        return torch.from_numpy(deq).view(self.master_layer_ids.shape).to(layer_ids_buf.device)

    def _ensure_cache(self):
        if not self.use_codebook:
            return
        if self._cache_valid:
            return
        self._master_weight_cache = self._dequantize_weight(
            self.master_layer_ids, self.master_code_ids, self.master_codebook
        )
        self._delta_weight_cache = self._dequantize_weight(
            self.delta_layer_ids, self.delta_code_ids, self.delta_codebook
        )
        self._cache_valid = True

    @property
    def effective_weight(self):
        if self.use_codebook:
            self._ensure_cache()
            return self._master_weight_cache + self._delta_weight_cache
        else:
            return self.master_weight + self.delta_weight

    @property
    def effective_bias(self):
        if self.master_bias is None:
            return None
        return self.master_bias + self.delta_bias

    def forward(self, x):
        weight = self.effective_weight
        bias = self.effective_bias
        return F.linear(x, weight, bias)

    def apply_delta_decay(self):
        if self.use_codebook:
            # Delta is stored as codebook indices, decay requires dequantize -> modify -> re-quantize
            # For inference-only mode, skip decay
            return
        with torch.no_grad():
            self.delta_weight.mul_(1.0 - self.delta_decay)
            if self.delta_bias is not None:
                self.delta_bias.mul_(1.0 - self.delta_decay)
            self.delta_weight.clamp_(-self.delta_clip, self.delta_clip)
            if self.delta_bias is not None:
                self.delta_bias.clamp_(-self.delta_clip, self.delta_clip)
            self._cache_valid = False

    def consolidate_delta(self):
        if self.use_codebook:
            # Merge delta into master by re-quantizing combined weights
            self._ensure_cache()
            combined = self._master_weight_cache + self._delta_weight_cache
            # Re-fit master codebook with combined values
            self.master_codebook.fit_from_data(combined.detach().cpu().numpy())
            m_layer_ids, m_code_ids = self.master_codebook.quantize(combined.detach().cpu().numpy())
            self.master_layer_ids.copy_(torch.from_numpy(m_layer_ids.reshape(self.master_layer_ids.shape)).int())
            self.master_code_ids.copy_(torch.from_numpy(m_code_ids.reshape(self.master_code_ids.shape)).int())
            # Reset delta
            self.delta_layer_ids.zero_()
            self.delta_code_ids.zero_()
            # Reset delta codebook to zero
            # For simplicity, keep delta codebook as is
            self._cache_valid = False
            return
        with torch.no_grad():
            self.master_weight.add_(self.delta_weight)
            if self.master_bias is not None:
                self.master_bias.add_(self.delta_bias)
            self.delta_weight.zero_()
            if self.delta_bias is not None:
                self.delta_bias.zero_()
            self._cache_valid = False

# Convenience wrapper to replace existing DualWeightLinear with codebook version
def make_codebook_dual_weight_layer(module):
    """
    Convert an existing DualWeightLinear module to codebook lazy mode in-place.
    Returns a new CodebookDualWeightLinear with weights compressed.
    """
    from dual_weight_layers import DualWeightLinear
    if not isinstance(module, DualWeightLinear):
        raise TypeError("module must be DualWeightLinear")
    new = CodebookDualWeightLinear(
        in_features=module.in_features,
        out_features=module.out_features,
        bias=module.master_bias is not None,
        delta_decay=getattr(module, 'delta_decay', 0.01),
        delta_clip=getattr(module, 'delta_clip', 1.0),
        use_codebook=True,
        codebook_capacity=1024,
    )
    # Copy bias
    if module.master_bias is not None:
        new.master_bias.data.copy_(module.master_bias.data)
        new.delta_bias.data.copy_(module.delta_bias.data)
    new.convert_from_weights(module.master_weight, module.delta_weight)
    return new

if __name__ == "__main__":
    # Quick sanity test
    lin = CodebookDualWeightLinear(32, 64, bias=True, use_codebook=True)
    # Initialize with random weights by converting from dummy
    dummy_master = torch.randn(64,32)
    dummy_delta = torch.randn(64,32)*0.01
    lin.convert_from_weights(dummy_master, dummy_delta)
    x = torch.randn(2,32)
    y = lin(x)
    print("Output shape:", y.shape)
    print("Codebook layers master:", len(lin.master_codebook.layers))
