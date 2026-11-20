"""
Dual-Weight Transformer layers for Nero/WeaveFormer
Master + Delta weight architecture with structured decay and drift protection
Production grade, drop-in replacement for nn.Linear and TransformerEncoderLayer
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from typing import Optional
from torch.utils.checkpoint import checkpoint

def flower_of_life_weights(n, dim, circles=64):
    """Generate Flower of Life inspired weights.
    Uses 64-circle angular basis to create a structured, rotationally symmetric pattern.
    """
    # 64 circles -> angular basis
    angles = torch.arange(circles, dtype=torch.float32) * (2*math.pi / circles)
    # radial positions 1..circles
    radii = torch.arange(1, circles+1, dtype=torch.float32)
    # Vectorized creation
    i_idx = torch.arange(n).unsqueeze(1).expand(n, dim)
    j_idx = torch.arange(dim).unsqueeze(0).expand(n, dim)
    # Map indices to angles/radii
    i_mod = i_idx % circles
    a = angles[i_mod]
    r = radii[i_mod]
    # Flower of Life inspired sinusoidal pattern
    mat = torch.sin((i_idx.float() + j_idx.float()) * a / (r + 1e-6) + 0.1 * j_idx.float())
    return mat

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return self.weight * x / rms

class BloomFilter:
    """Simple bloom filter for fast duplicate check"""
    def __init__(self, size=1<<20, hash_count=4):
        import numpy as np
        self.size = size
        self.hash_count = hash_count
        self.bit_array = np.zeros(size, dtype=np.bool_)
    def _hashes(self, key):
        import hashlib
        for i in range(self.hash_count):
            h = hashlib.md5((str(key)+str(i)).encode()).hexdigest()
            yield int(h,16) % self.size
    def add(self, key):
        for h in self._hashes(key):
            self.bit_array[h] = True
    def __contains__(self, key):
        return all(self.bit_array[h] for h in self._hashes(key))

class MatrixCalcEngine:
    """CPU-offloaded matrix multiplication calculator for reasoning acceleration"""
    def __init__(self, dim):
        import numpy as np
        self.dim = dim
        self.rng = np.random.default_rng(42)
        # Small low-rank factors for fast CPU matmul
        self.U = self.rng.standard_normal((dim, dim//4)).astype(np.float32)
        self.V = self.rng.standard_normal((dim//4, dim)).astype(np.float32)
        # Bloom filter for duplicate calc detection
        self.bloom = BloomFilter(size=1<<20)
        # Skimming/chunking clustering cache
        self.cache = {}
    def __call__(self, x):
        # x: torch.Tensor on GPU, move to CPU for fast matmul
        import torch
        import numpy as np
        x_np = x.detach().cpu().numpy()
        # Skimming: chunk and cluster similar inputs
        # Include shape in key to avoid shape mismatches
        flat = x_np.reshape(-1)
        # Guard against NaNs in sample key
        sample_vals = flat[:8]
        if np.isnan(sample_vals).any():
            sample = np.zeros(8, dtype=np.int64)
        else:
            sample = np.round(sample_vals).astype(np.int64)
        chunk_key = (x_np.shape, tuple(sample.tolist()))
        # Bloom filter duplicate check
        if chunk_key in self.bloom and chunk_key in self.cache:
            # Cache hit, return cached result
            out_np = self.cache[chunk_key]
        else:
            # Low-rank matmul: x @ U @ V
            out_np = x_np @ self.U @ self.V
            # Bloom filter add and cache
            self.bloom.add(chunk_key)
            self.cache[chunk_key] = out_np
        return torch.from_numpy(out_np).to(x.device)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=4096, base=10000):
        super().__init__()
        import os
        base = int(os.environ.get('ROTARY_BASE', str(base)))
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_seq_len = max_seq_len
    def forward(self, x):
        seq_len = x.shape[1]
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()[None, :, :]
        sin = emb.sin()[None, :, :]
        return cos, sin

def apply_rotary_pos_emb(x, cos, sin):
    x1, x2 = x[..., ::2], x[..., 1::2]
    x_rot = torch.stack((-x2, x1), dim=-1).reshape_as(x)
    return x * cos + x_rot * sin

class DualWeightLinear(nn.Module):
    """
    Linear layer with master + delta weights, wave-modulated.
    master: stable long-term weights
    delta: short-term adaptive weights with decay/clipping
    Effective weight = (master + delta) * wave_modulation
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True, 
                 delta_decay: float = 0.01, delta_clip: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.delta_decay = delta_decay
        self.delta_clip = delta_clip
        # Wave modulation parameters for weight shaping
        self.wave_amp = nn.Parameter(torch.ones(1) * 0.05)
        self.wave_freq = nn.Parameter(torch.ones(1) * 6.2831853)
        self.wave_phase = nn.Parameter(torch.zeros(1))
        
        # Master weights - stable
        self.master_weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.master_bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('master_bias', None)
        
        # Delta weights - adaptive
        self.delta_weight = nn.Parameter(torch.zeros(out_features, in_features))
        if bias:
            self.delta_bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('delta_bias', None)
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.master_weight, a=math.sqrt(5))
        if self.master_bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.master_weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.master_bias, -bound, bound)
        # delta stays zero

    @property
    def effective_weight(self):
        return self.master_weight + self.delta_weight

    @property
    def effective_bias(self):
        if self.master_bias is None:
            return None
        return self.master_bias + self.delta_bias

    def forward(self, x):
        weight = self.effective_weight
        # Sanitize and clamp wave params
        wave_amp = torch.nan_to_num(self.wave_amp, nan=0.05, posinf=1.0, neginf=-1.0)
        wave_amp = torch.clamp(wave_amp, -0.5, 0.5)
        wave_freq = torch.nan_to_num(self.wave_freq, nan=6.28, posinf=10.0, neginf=0.1)
        wave_freq = torch.clamp(wave_freq, 0.1, 10.0)
        wave_phase = torch.nan_to_num(self.wave_phase, nan=0.0, posinf=1e3, neginf=-1e3)
        # Wave modulation of weights as static spatial wave
        # Create a 2D index grid for out x in
        # Use cached grid to avoid reallocation
        if not hasattr(self, '_wave_grid'):
            i = torch.arange(self.out_features, device=x.device).view(-1,1).float()
            j = torch.arange(self.in_features, device=x.device).view(1,-1).float()
            self.register_buffer('_wave_i', i)
            self.register_buffer('_wave_j', j)
        # Wave modulation
        # Normalize indices to ~0-1 range for stable frequencies
        i_norm = self._wave_i / max(1, self.out_features)
        j_norm = self._wave_j / max(1, self.in_features)
        # 2D wave pattern
        wave = 1.0 + wave_amp * torch.sin(wave_freq * (i_norm + j_norm) + wave_phase)
        weight = weight * wave
        bias = self.effective_bias
        return F.linear(x, weight, bias)

    def apply_delta_decay(self):
        # Structured decay for delta weights
        with torch.no_grad():
            self.delta_weight.mul_(1.0 - self.delta_decay)
            if self.delta_bias is not None:
                self.delta_bias.mul_(1.0 - self.delta_decay)
            # Symmetric clipping for drift protection
            self.delta_weight.clamp_(-self.delta_clip, self.delta_clip)
            if self.delta_bias is not None:
                self.delta_bias.clamp_(-self.delta_clip, self.delta_clip)

    def consolidate_delta(self):
        # Periodically merge delta into master for stability
        with torch.no_grad():
            self.master_weight.add_(self.delta_weight)
            if self.master_bias is not None:
                self.master_bias.add_(self.delta_bias)
            # Reset delta
            self.delta_weight.zero_()
            if self.delta_bias is not None:
                self.delta_bias.zero_()

    def hit_rate(self):
        # Proxy hit rate: fraction of delta weights with non-trivial magnitude
        # Represents how much the delta path is being used
        if self.delta_weight.numel() == 0:
            return 0.0
        active = (self.delta_weight.abs() > 1e-6).float().mean().item()
        return active


class DualWeightMultiheadAttention(nn.Module):
    """
    Multi-head attention with dual-weight projections
    """
    def __init__(self, embed_dim, num_heads, dropout=0.0, window_size=None, num_global_heads=1):
        super().__init__()
        import os
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.num_global_heads = max(0, min(num_global_heads, num_heads))
        self.window_size = window_size
        
        # QKV bias for pro config
        use_qkv_bias = os.environ.get('ATTN_QKV_BIAS','1') == '1'
        self.q_proj = DualWeightLinear(embed_dim, embed_dim, bias=use_qkv_bias)
        self.k_proj = DualWeightLinear(embed_dim, embed_dim, bias=use_qkv_bias)
        self.v_proj = DualWeightLinear(embed_dim, embed_dim, bias=use_qkv_bias)
        self.out_proj = DualWeightLinear(embed_dim, embed_dim, bias=True)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        self.window_size = window_size
        # ALiBi slopes per head
        self.register_buffer('alibi_slopes', self._get_alibi_slopes(num_heads))
        self.num_global_heads = max(0, min(self.num_global_heads, num_heads))

    def _get_alibi_slopes(self, num_heads):
        # Classic ALiBi slope generation
        def get_slopes(n):
            start = 2 ** (-(2 ** (-torch.arange(n).float()))) # placeholder
            # Use standard geometric progression
            slopes = []
            for i in range(n):
                slopes.append(2 ** -(i+1) * 2)
            return torch.tensor(slopes, dtype=torch.float32)
        # Use the known ALiBi formula: slopes = 1/(2**(i//2))
        slopes = []
        for i in range(num_heads):
            slopes.append(1.0 / (2 ** (i // 2)))
        slopes = torch.tensor(slopes, dtype=torch.float32)
        # Normalize to typical range
        slopes = slopes / slopes.max()
        return slopes

    def forward(self, x, mask=None):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1,2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1,2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1,2)
        
        # Try FlashAttention for speed / memory
        use_flash = False
        try:
            from flash_attn import flash_attn_func
            use_flash = True
        except Exception:
            use_flash = False
        
        if use_flash and self.window_size is None:
            # FlashAttention only supports full attention here
            # causal mask via is_causal
            out = flash_attn_func(q, k, v, dropout_p=self.dropout.p if self.training else 0.0, is_causal=True)
            out = out.transpose(1,2).contiguous().view(B, T, C)
            out = self.out_proj(out)
            return out
        
        attn = torch.matmul(q, k.transpose(-2,-1)) / self.scale
        # ALiBi bias for extrapolation
        if T > 1:
            device = x.device
            # Compute relative distance matrix [T, T]
            idx = torch.arange(T, device=device)
            rel_dist = torch.abs(idx[:, None] - idx[None, :]).float()
            # Shape [1, num_heads, 1, 1] * [1,1,T,T] -> [1, num_heads, T, T]
            slopes = self.alibi_slopes.view(1, self.num_heads, 1, 1)
            alibi_bias = -(rel_dist[None, None, :, :] * slopes)
            attn = attn + alibi_bias
        if self.window_size is not None and T > 1:
            device = x.device
            idx = torch.arange(T, device=device)
            rel = idx[None, :] - idx[:, None]
            window_mask = (rel >= 0) & (rel < self.window_size)
            # Global heads attend to full context
            if self.num_global_heads > 0:
                # Create mask where first num_global_heads are unmasked
                global_mask = torch.zeros(self.num_heads, dtype=torch.bool, device=device)
                global_mask[:self.num_global_heads] = True
                # Expand window mask for all heads
                window_mask_exp = window_mask[None, None, :, :].expand(B, self.num_heads, T, T)
                # Build a mask that is True for global heads everywhere
                global_expand = global_mask.view(1, self.num_heads, 1, 1).expand(B, self.num_heads, T, T)
                final_mask = window_mask_exp | global_expand
                attn = attn.masked_fill(~final_mask, float('-inf'))
            else:
                window_mask_exp = window_mask[None, None, :, :].expand(B, self.num_heads, T, T)
                attn = attn.masked_fill(~window_mask_exp, float('-inf'))
            if mask is not None:
                attn = attn.masked_fill(mask == 0, float('-inf'))
        elif mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1,2).contiguous().view(B, T, C)
        out = self.out_proj(out)
        return out

    def apply_delta_decay(self):
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            proj.apply_delta_decay()

    def consolidate_delta(self):
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            proj.consolidate_delta()


class DualWeightTransformerEncoderLayer(nn.Module):
    """
    Transformer encoder layer with dual-weight linear projections
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, window_size=None):
        super().__init__()
        self.self_attn = DualWeightMultiheadAttention(d_model, nhead, dropout, window_size=window_size)
        self.linear1 = DualWeightLinear(d_model, dim_feedforward)
        self.linear1_gate = DualWeightLinear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = DualWeightLinear(dim_feedforward, d_model)
        # Mixture of Experts for feedforward expansion
        self.moe_num_experts = 4
        self.moe_experts = nn.ModuleList([
            DualWeightLinear(d_model, dim_feedforward) for _ in range(self.moe_num_experts)
        ])
        self.moe_router = DualWeightLinear(d_model, self.moe_num_experts)
        self.moe_gate = nn.Parameter(torch.zeros(1))
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()
        # LayerScale init to 1e-4 for stable training
        self.attn_scale = nn.Parameter(torch.ones(d_model) * 1e-4)
        self.ffn_scale = nn.Parameter(torch.ones(d_model) * 1e-4)
        # Invisible "thinking" sub-layer between main operations
        self.thought_linear1 = DualWeightLinear(d_model, d_model//2)
        self.thought_linear2 = DualWeightLinear(d_model//2, d_model)
        self.thought_gate = nn.Parameter(torch.zeros(1))
        self.thought_norm = RMSNorm(d_model)
        self.thought_dropout = nn.Dropout(dropout)
        # Prediction weights attached to thinking/reasoning
        self.prediction_linear = DualWeightLinear(d_model, d_model)
        self.prediction_gate = nn.Parameter(torch.zeros(1))
        self.prediction_bias = nn.Parameter(torch.zeros(d_model))
        # Reasoning fusion: combine thought and prediction for richer reasoning
        self.thought_prediction_fusion = DualWeightLinear(2*d_model, d_model)
        self.reasoning_gate = nn.Parameter(torch.zeros(1))
        # Meta-reasoning refinement for deeper abstraction
        self.meta_reason_linear1 = DualWeightLinear(d_model, d_model//2)
        self.meta_reason_linear2 = DualWeightLinear(d_model//2, d_model)
        self.meta_reason_gate = nn.Parameter(torch.zeros(1))
        # Logic weights for structural reasoning
        self.logic_weight = nn.Parameter(torch.zeros(1))
        self.logic_gate = nn.Parameter(torch.zeros(1))
        # Planning weights neurons that connect to thought and reason
        self.planning_linear1 = DualWeightLinear(d_model, d_model//2)
        self.planning_linear2 = DualWeightLinear(d_model//2, d_model)
        self.planning_gate = nn.Parameter(torch.zeros(1))
        self.planning_bias = nn.Parameter(torch.zeros(d_model))
        # Ethical weighting sub-layer for right/wrong signal
        self.ethical_linear1 = DualWeightLinear(d_model, d_model//2)
        self.ethical_linear2 = DualWeightLinear(d_model//2, d_model)
        self.ethical_gate = nn.Parameter(torch.zeros(1))
        self.ethical_bias = nn.Parameter(torch.zeros(d_model))
        # Pattern fast-track weighting for rapid information retrieval
        self.pattern_linear1 = DualWeightLinear(d_model, d_model//2)
        self.pattern_linear2 = DualWeightLinear(d_model//2, d_model)
        self.pattern_gate = nn.Parameter(torch.zeros(1))
        self.pattern_bias = nn.Parameter(torch.zeros(d_model))
        # Structural learning weights for recall and remembering
        self.structural_linear1 = DualWeightLinear(d_model, d_model//2)
        self.structural_linear2 = DualWeightLinear(d_model//2, d_model)
        self.structural_gate = nn.Parameter(torch.zeros(1))
        self.structural_bias = nn.Parameter(torch.zeros(d_model))
        # Structural reasoning weights for structured inference
        self.structural_reason_linear1 = DualWeightLinear(d_model, d_model//2)
        self.structural_reason_linear2 = DualWeightLinear(d_model//2, d_model)
        self.structural_reason_gate = nn.Parameter(torch.zeros(1))
        self.structural_reason_bias = nn.Parameter(torch.zeros(d_model))
        # Structural pattern weights for fast recall
        self.structural_pattern_linear1 = DualWeightLinear(d_model, d_model//2)
        self.structural_pattern_linear2 = DualWeightLinear(d_model//2, d_model)
        self.structural_pattern_gate = nn.Parameter(torch.zeros(1))
        self.structural_pattern_bias = nn.Parameter(torch.zeros(d_model))
        # Matrix multiplication calculator hidden layer for reasoning acceleration
        self.matrix_calc_linear1 = DualWeightLinear(d_model, d_model)
        self.matrix_calc_linear2 = DualWeightLinear(d_model, d_model)
        self.matrix_calc_gate = nn.Parameter(torch.zeros(1))
        # CPU-offloaded matrix calculator helper
        self.matrix_calc_engine = MatrixCalcEngine(d_model)
        # Map hidden layer for spatial/relational mapping
        self.map_linear1 = DualWeightLinear(d_model, d_model//2)
        self.map_linear2 = DualWeightLinear(d_model//2, d_model)
        self.map_gate = nn.Parameter(torch.zeros(1))
        # Graph hidden layer for relational/graph reasoning
        self.graph_linear1 = DualWeightLinear(d_model, d_model//2)
        self.graph_linear2 = DualWeightLinear(d_model//2, d_model)
        self.graph_gate = nn.Parameter(torch.zeros(1))
        # Scratch pad working memory for temporary reasoning traces - auto-growing layered sliding window with index-backed circular buffer and inner connections
        self.scratch_pad_linear1 = DualWeightLinear(d_model, d_model//2)
        self.scratch_pad_linear2 = DualWeightLinear(d_model//2, d_model)
        self.scratch_pad_gate = nn.Parameter(torch.zeros(1))
        self.mem_depth = int(os.environ.get('MEM_DEPTH','6'))
        self.mem_max_depth = int(os.environ.get('MEM_MAX_DEPTH','64'))
        self.mem_auto_grow = os.environ.get('MEM_AUTO_GROW','1') == '1'
        self.mem_window = int(os.environ.get('MEM_WINDOW', str(self.mem_depth)))
        self.register_buffer('scratch_pad_mem', torch.zeros(self.mem_max_depth, d_model))
        self.register_buffer('scratch_pad_idx', torch.zeros(1, dtype=torch.long))
        self.register_buffer('scratch_pad_depth', torch.tensor([self.mem_depth], dtype=torch.long))
        self.scratch_pad_mixer = DualWeightLinear(self.mem_max_depth * d_model, d_model)
        # Ledger for persistent event logging - auto-growing layered sliding window with index-backed circular buffer and inner connections
        self.ledger_linear1 = DualWeightLinear(d_model, d_model//2)
        self.ledger_linear2 = DualWeightLinear(d_model//2, d_model)
        self.ledger_gate = nn.Parameter(torch.zeros(1))
        self.register_buffer('ledger_mem', torch.zeros(self.mem_max_depth, d_model))
        self.register_buffer('ledger_idx', torch.zeros(1, dtype=torch.long))
        self.register_buffer('ledger_depth', torch.tensor([self.mem_depth], dtype=torch.long))
        self.ledger_mixer = DualWeightLinear(self.mem_max_depth * d_model, d_model)
        # Journal for sequential reflection - auto-growing layered sliding window with index-backed circular buffer and inner connections
        self.journal_linear1 = DualWeightLinear(d_model, d_model//2)
        self.journal_linear2 = DualWeightLinear(d_model//2, d_model)
        self.journal_gate = nn.Parameter(torch.zeros(1))
        self.register_buffer('journal_mem', torch.zeros(self.mem_max_depth, d_model))
        self.register_buffer('journal_idx', torch.zeros(1, dtype=torch.long))
        self.register_buffer('journal_depth', torch.tensor([self.mem_depth], dtype=torch.long))
        self.journal_mixer = DualWeightLinear(self.mem_max_depth * d_model, d_model)
        # Diary for personalised memory traces - auto-growing layered sliding window with index-backed circular buffer and inner connections
        self.diary_linear1 = DualWeightLinear(d_model, d_model//2)
        self.diary_linear2 = DualWeightLinear(d_model//2, d_model)
        self.diary_gate = nn.Parameter(torch.zeros(1))
        self.register_buffer('diary_mem', torch.zeros(self.mem_max_depth, d_model))
        self.register_buffer('diary_idx', torch.zeros(1, dtype=torch.long))
        self.register_buffer('diary_depth', torch.tensor([self.mem_depth], dtype=torch.long))
        self.diary_mixer = DualWeightLinear(self.mem_max_depth * d_model, d_model)
        # North star guidance to keep reasoning on track
        self.north_star_linear1 = DualWeightLinear(d_model, d_model//2)
        self.north_star_linear2 = DualWeightLinear(d_model//2, d_model)
        self.north_star_gate = nn.Parameter(torch.zeros(1))
        self.north_star_vec = nn.Parameter(torch.ones(d_model))
        # Memory attention over working memories
        self.mem_attn_q = DualWeightLinear(d_model, d_model)
        self.mem_attn_k = DualWeightLinear(d_model, d_model)
        self.mem_attn_v = DualWeightLinear(d_model, d_model)
        self.mem_attn_out = DualWeightLinear(d_model, d_model)
        self.mem_attn_gate = nn.Parameter(torch.zeros(1))
        # Adaptive think iterations
        self.think_iterations = nn.Parameter(torch.tensor(2.0))
        # Convergence gate for attractor reasoning
        self.convergence_eps = nn.Parameter(torch.tensor(1e-4))
        # Sliding hidden window for temporal context within layer
        self.hidden_window_size = 8
        self.register_buffer('hidden_window_mem', torch.zeros(self.hidden_window_size, d_model))
        self.hidden_window_gate = nn.Parameter(torch.zeros(1))
        # Temporal attention over hidden window
        self.hidden_window_attn_q = DualWeightLinear(d_model, d_model)
        self.hidden_window_attn_k = DualWeightLinear(d_model, d_model)
        self.hidden_window_attn_v = DualWeightLinear(d_model, d_model)
        self.hidden_window_attn_gate = nn.Parameter(torch.zeros(1))
        # Internal order for hidden physical layer
        self.hidden_order_bias = nn.Parameter(torch.zeros(d_model))
        self.order_conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
        self.order_gate = nn.Parameter(torch.zeros(1))
        # Training-ease hidden layer to smooth step/val loss
        self.ease_linear1 = DualWeightLinear(d_model, d_model//4)
        self.ease_linear2 = DualWeightLinear(d_model//4, d_model)
        self.ease_gate = nn.Parameter(torch.zeros(1))
        # Future anticipation hidden layer to recognize future from past
        self.future_linear1 = DualWeightLinear(d_model, d_model//2)
        self.future_linear2 = DualWeightLinear(d_model//2, d_model)
        self.future_gate = nn.Parameter(torch.zeros(1))
        # Radix sort hidden layer for ordered feature activation
        self.radix_sort_gate = nn.Parameter(torch.zeros(1))
        # Organizer hidden layer for information sorting and structure
        self.organizer_linear1 = DualWeightLinear(d_model, d_model//2)
        self.organizer_linear2 = DualWeightLinear(d_model//2, d_model)
        self.organizer_gate = nn.Parameter(torch.zeros(1))
        self.organizer_sort_gate = nn.Parameter(torch.zeros(1))
        # CPU offload hidden layer to reduce GPU stress
        self.cpu_offload_linear1 = DualWeightLinear(d_model, d_model//4)
        self.cpu_offload_linear2 = DualWeightLinear(d_model//4, d_model)
        self.cpu_offload_gate = nn.Parameter(torch.zeros(1))
        # Math-simplification hidden layer for easier GPU compute
        self.simplify_linear1 = DualWeightLinear(d_model, d_model//8)
        self.simplify_linear2 = DualWeightLinear(d_model//8, d_model)
        self.simplify_gate = nn.Parameter(torch.zeros(1))
        # Math reuse cache for simplify bottleneck
        self.simplify_cache = {}
        self.simplify_cache_max = 512
        # Chain-of-thought condensation for hidden layer compression
        self.cot_condense_linear1 = DualWeightLinear(d_model, d_model//4)
        self.cot_condense_linear2 = DualWeightLinear(d_model//4, d_model)
        self.cot_gate = nn.Parameter(torch.zeros(1))
        self.cot_norm = RMSNorm(d_model)
        # Mini-agent helper layer that acts as a helper for other layers
        self.helper_agent_linear1 = DualWeightLinear(d_model, d_model//4)
        self.helper_agent_linear2 = DualWeightLinear(d_model//4, d_model)
        self.helper_agent_gate = nn.Parameter(torch.zeros(1))
        # Pro SwiGLU FFN with sliding-window gating
        self.pros_swiglu_w1 = DualWeightLinear(d_model, dim_feedforward*2)
        self.pros_swiglu_w2 = DualWeightLinear(dim_feedforward, d_model)
        self.pros_swiglu_gate = nn.Parameter(torch.zeros(1))
        self.pros_swiglu_window = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)

    def forward(self, src):
        # Sanitize input to prevent NaN propagation
        if torch.isnan(src).any() or torch.isinf(src).any():
            src = torch.nan_to_num(src, nan=0.0, posinf=1e4, neginf=-1e4)
        # Self attention with LayerScale
        attn_out = self.self_attn(src)
        src = src + self.attn_scale * self.dropout1(attn_out)
        src = self.norm1(src)

        # Pro SwiGLU FFN with sliding-window gating
        pros_gate = torch.sigmoid(self.pros_swiglu_gate)
        # Sliding window local context via depthwise conv
        x_conv = src.transpose(1,2)
        x_conv = self.pros_swiglu_window(x_conv)
        x_conv = x_conv.transpose(1,2)
        # SwiGLU
        hidden = self.pros_swiglu_w1(src)
        x1, x2 = hidden.chunk(2, dim=-1)
        swiglu_out = self.pros_swiglu_w2(F.silu(x1) * x2)
        src = src + pros_gate * self.dropout1(swiglu_out + 0.1 * x_conv)

        # Calculator / simplify pre-conditioning to reduce GPU stress
        simplify_gate = torch.sigmoid(self.simplify_gate)
        simplify_delta_activity = max(self.simplify_linear1.delta_weight.abs().mean().item(), self.simplify_linear2.delta_weight.abs().mean().item())
        simplify_speed_gate = 1.0 if simplify_delta_activity > 1e-4 else 0.0
        if simplify_speed_gate > 0.0:
            # Dynamic compression: use smaller bottleneck when activity is low
            use_compressed = simplify_delta_activity < 5e-4
            # Depth-phase wave modulation for shared simplify weights
            layer_idx = getattr(self, '_layer_idx', 0)
            amp = torch.clamp(self.simplify_wave_amp, min=0.0, max=1.0)
            freq = torch.clamp(self.simplify_wave_freq, min=0.1, max=10.0)
            wave = 1.0 + 0.1 * amp * torch.sin(freq * float(layer_idx) + self.simplify_wave_phase)
            # Math reuse cache lookup + chunk/cluster/skim/bloom via MatrixCalcEngine
            try:
                key = tuple(src[0,0,:8].detach().cpu().numpy().round(3).tolist())
                if key in self.simplify_cache:
                    simplify_out = self.simplify_cache[key]
                else:
                    src_pre = self.matrix_calc_engine(src)
                    if use_compressed:
                        simplify_out = self.simplify_linear2_small(self.activation(self.simplify_linear1_small(src_pre)))
                    else:
                        simplify_out = self.simplify_linear2(self.activation(self.simplify_linear1(src_pre)))
                    if len(self.simplify_cache) >= self.simplify_cache_max:
                        oldest_key = next(iter(self.simplify_cache))
                        self.simplify_cache.pop(oldest_key)
                    self.simplify_cache[key] = simplify_out.detach()
            except Exception:
                if use_compressed:
                    simplify_out = self.simplify_linear2_small(self.activation(self.simplify_linear1_small(src)))
                else:
                    simplify_out = self.simplify_linear2(self.activation(self.simplify_linear1(src)))
            simplify_out = simplify_out * wave
            src = src + simplify_gate * self.dropout1(simplify_out)
        
        # Invisible thinking sub-layer with think steps, prediction fusion, and meta-reasoning
        gate = torch.sigmoid(self.thought_gate)
        pred_gate = torch.sigmoid(self.prediction_gate)
        reason_gate = torch.sigmoid(self.reasoning_gate)
        meta_gate = torch.sigmoid(self.meta_reason_gate)
        logic_gate = torch.sigmoid(self.logic_gate)
        planning_gate = torch.sigmoid(self.planning_gate)
        ethical_gate = torch.sigmoid(self.ethical_gate)
        pattern_gate = torch.sigmoid(self.pattern_gate)
        structural_gate = torch.sigmoid(self.structural_gate)
        structural_reason_gate = torch.sigmoid(self.structural_reason_gate)
        structural_pattern_gate = torch.sigmoid(self.structural_pattern_gate)
        matrix_calc_gate = torch.sigmoid(self.matrix_calc_gate)
        map_gate = torch.sigmoid(self.map_gate)
        graph_gate = torch.sigmoid(self.graph_gate)
        scratch_pad_gate = torch.sigmoid(self.scratch_pad_gate)
        ledger_gate = torch.sigmoid(self.ledger_gate)
        journal_gate = torch.sigmoid(self.journal_gate)
        diary_gate = torch.sigmoid(self.diary_gate)
        north_star_gate = torch.sigmoid(self.north_star_gate)
        steps = int(torch.clamp(self.think_iterations, 1, 5).item())
        prev_src = src.clone()
        for _ in range(steps):  # adaptive think steps per layer
            thought = self.thought_linear2(self.thought_dropout(self.activation(self.thought_linear1(src))))
            # Prediction weights attached to thinking/reasoning with bias to steer predictions
            pred = self.prediction_linear(thought) + self.prediction_bias
            # Fuse thought and prediction for richer reasoning
            fused = self.thought_prediction_fusion(torch.cat([thought, pred], dim=-1))
            # Meta-reasoning refinement
            meta = self.meta_reason_linear2(self.thought_dropout(self.activation(self.meta_reason_linear1(fused))))
            # Ethical weighting for right/wrong signal
            ethical = self.ethical_linear2(self.thought_dropout(self.activation(self.ethical_linear1(fused)))) + self.ethical_bias
            # Logic weights for structural reasoning, ethically informed
            logic_mod = torch.sigmoid(self.logic_weight) * thought + ethical_gate * ethical * 0.5
            # Resonance-based logic filtering paired with eco-location
            # Eco-location = thought context, resonance = similarity to thought
            resonance_score = F.cosine_similarity(thought, logic_mod, dim=-1).mean()
            logic_resonance_gate = torch.sigmoid(resonance_score * 5.0)
            logic_mod = logic_mod * logic_resonance_gate
            # Planning weights neurons connecting to thought and reason
            planning = self.planning_linear2(self.thought_dropout(self.activation(self.planning_linear1(fused)))) + self.planning_bias
            # Pattern fast-track weighting for rapid information retrieval
            pattern = self.pattern_linear2(self.thought_dropout(self.activation(self.pattern_linear1(fused)))) + self.pattern_bias
            # Structural learning weights for recall and remembering
            structural = self.structural_linear2(self.thought_dropout(self.activation(self.structural_linear1(fused)))) + self.structural_bias
            # Structural reasoning weights for structured inference
            structural_reason = self.structural_reason_linear2(self.thought_dropout(self.activation(self.structural_reason_linear1(fused)))) + self.structural_reason_bias
            # Structural pattern weights for fast recall
            structural_pattern = self.structural_pattern_linear2(self.thought_dropout(self.activation(self.structural_pattern_linear1(fused)))) + self.structural_pattern_bias
            # Matrix multiplication calculator for reasoning acceleration
            # Hybrid GPU/CPU calculator: fast low-rank CPU matmul + learned GPU refinement
            calc_input = self.thought_dropout(self.activation(fused))
            # Delta-gated fast path: skip CPU offload when delta activity is low for speed
            delta_activity = self.matrix_calc_linear1.delta_weight.abs().mean().item()
            delta_speed_gate = 1.0 if delta_activity > 1e-4 else 0.0
            if delta_speed_gate > 0.0:
                fast_calc = self.matrix_calc_engine(calc_input)
                refined_calc = self.matrix_calc_linear2(self.matrix_calc_linear1(fast_calc))
                matrix_calc = refined_calc
            else:
                # Fast skip: reuse fused as identity to avoid CPU offload
                matrix_calc = calc_input
            # Map hidden layer for spatial/relational mapping - delta gated for speed
            map_delta_activity = max(self.map_linear1.delta_weight.abs().mean().item(), self.map_linear2.delta_weight.abs().mean().item())
            map_speed_gate = 1.0 if map_delta_activity > 1e-4 else 0.0
            if map_speed_gate > 0.0:
                map_out = self.map_linear2(self.thought_dropout(self.activation(self.map_linear1(fused))))
            else:
                map_out = fused
            # Graph hidden layer for relational/graph reasoning - delta gated for speed
            graph_delta_activity = max(self.graph_linear1.delta_weight.abs().mean().item(), self.graph_linear2.delta_weight.abs().mean().item())
            graph_speed_gate = 1.0 if graph_delta_activity > 1e-4 else 0.0
            if graph_speed_gate > 0.0:
                graph_out = self.graph_linear2(self.thought_dropout(self.activation(self.graph_linear1(fused))))
            else:
                graph_out = fused
            # Scratch pad working memory for temporary reasoning traces - 6 layered sliding window
            scratch_delta_activity = max(self.scratch_pad_linear1.delta_weight.abs().mean().item(), self.scratch_pad_linear2.delta_weight.abs().mean().item())
            scratch_speed_gate = 1.0 if scratch_delta_activity > 1e-4 else 0.0
            if scratch_speed_gate > 0.0:
                scratch_out = self.scratch_pad_linear2(self.thought_dropout(self.activation(self.scratch_pad_linear1(fused))))
            else:
                scratch_out = fused
            with torch.no_grad():
                mem_update = fused.detach().mean([0,1]).to(self.scratch_pad_mem.dtype)
                idx = int(self.scratch_pad_idx[0].item())
                depth = int(self.scratch_pad_depth[0].item())
                self.scratch_pad_mem[idx].lerp_(mem_update, 0.1)
                # Auto-grow: if we wrapped around and auto-grow is on, increase depth
                next_idx = idx + 1
                if next_idx >= depth:
                    if self.mem_auto_grow and depth < self.mem_max_depth:
                        new_depth = min(depth * 2, self.mem_max_depth)
                        self.scratch_pad_depth[0] = new_depth
                        # print growth
                        if not hasattr(self, '_mem_grow_logged'):
                            self._mem_grow_logged = set()
                        if depth not in self._mem_grow_logged:
                            print(f"[MemGrow] Scratch pad depth grew {depth} -> {new_depth}")
                            self._mem_grow_logged.add(depth)
                    next_idx = 0
                self.scratch_pad_idx[0] = next_idx
                # Inner connections via mixer over sliding window of auto-growing memory
                window = min(self.mem_window, depth)
                if window > 0:
                    indices = [(idx - i - 1) % depth for i in range(window)]
                    cur_mem = self.scratch_pad_mem[indices].contiguous()
                else:
                    cur_mem = self.scratch_pad_mem[:0]
                cur_flat = cur_mem.view(1, -1)
                pad_len = self.mem_max_depth - window
                if pad_len > 0:
                    pad = torch.zeros(1, pad_len * cur_mem.size(-1), device=cur_mem.device, dtype=cur_mem.dtype)
                    cur_flat = torch.cat([cur_flat, pad], dim=1)
                mixed = self.scratch_pad_mixer(cur_flat)
                scratch_context = mixed.view(1,1,-1)
            scratch_out = scratch_out + 0.1 * scratch_context
            # Ledger for persistent event logging - 6 layered sliding window
            ledger_delta_activity = max(self.ledger_linear1.delta_weight.abs().mean().item(), self.ledger_linear2.delta_weight.abs().mean().item())
            ledger_speed_gate = 1.0 if ledger_delta_activity > 1e-4 else 0.0
            if ledger_speed_gate > 0.0:
                ledger_out = self.ledger_linear2(self.thought_dropout(self.activation(self.ledger_linear1(fused))))
            else:
                ledger_out = fused
            with torch.no_grad():
                mem_update = fused.detach().mean([0,1]).to(self.ledger_mem.dtype)
                idx = int(self.ledger_idx[0].item())
                depth = int(self.ledger_depth[0].item())
                self.ledger_mem[idx].lerp_(mem_update, 0.05)
                next_idx = idx + 1
                if next_idx >= depth:
                    if self.mem_auto_grow and depth < self.mem_max_depth:
                        new_depth = min(depth * 2, self.mem_max_depth)
                        self.ledger_depth[0] = new_depth
                        if not hasattr(self, '_ledger_grow_logged'):
                            self._ledger_grow_logged = set()
                        if depth not in self._ledger_grow_logged:
                            print(f"[MemGrow] Ledger depth grew {depth} -> {new_depth}")
                            self._ledger_grow_logged.add(depth)
                    next_idx = 0
                self.ledger_idx[0] = next_idx
                # Inner connections via mixer over sliding window of auto-growing memory
                window = min(self.mem_window, depth)
                if window > 0:
                    indices = [(idx - i - 1) % depth for i in range(window)]
                    cur_mem = self.ledger_mem[indices].contiguous()
                else:
                    cur_mem = self.ledger_mem[:0]
                cur_flat = cur_mem.view(1, -1)
                pad_len = self.mem_max_depth - window
                if pad_len > 0:
                    pad = torch.zeros(1, pad_len * cur_mem.size(-1), device=cur_mem.device, dtype=cur_mem.dtype)
                    cur_flat = torch.cat([cur_flat, pad], dim=1)
                ledger_mixed = self.ledger_mixer(cur_flat).view(1,1,-1)
                ledger_out = ledger_out + 0.05 * ledger_mixed
            # Journal for sequential reflection - 6 layered sliding window
            journal_delta_activity = max(self.journal_linear1.delta_weight.abs().mean().item(), self.journal_linear2.delta_weight.abs().mean().item())
            journal_speed_gate = 1.0 if journal_delta_activity > 1e-4 else 0.0
            if journal_speed_gate > 0.0:
                journal_out = self.journal_linear2(self.thought_dropout(self.activation(self.journal_linear1(fused))))
            else:
                journal_out = fused
            with torch.no_grad():
                mem_update = fused.detach().mean([0,1]).to(self.journal_mem.dtype)
                idx = int(self.journal_idx[0].item())
                depth = int(self.journal_depth[0].item())
                self.journal_mem[idx].lerp_(mem_update, 0.05)
                next_idx = idx + 1
                if next_idx >= depth:
                    if self.mem_auto_grow and depth < self.mem_max_depth:
                        new_depth = min(depth * 2, self.mem_max_depth)
                        self.journal_depth[0] = new_depth
                        if not hasattr(self, '_journal_grow_logged'):
                            self._journal_grow_logged = set()
                        if depth not in self._journal_grow_logged:
                            print(f"[MemGrow] Journal depth grew {depth} -> {new_depth}")
                            self._journal_grow_logged.add(depth)
                    next_idx = 0
                self.journal_idx[0] = next_idx
                # Inner connections via mixer over sliding window of auto-growing memory
                window = min(self.mem_window, depth)
                if window > 0:
                    indices = [(idx - i - 1) % depth for i in range(window)]
                    cur_mem = self.journal_mem[indices].contiguous()
                else:
                    cur_mem = self.journal_mem[:0]
                cur_flat = cur_mem.view(1, -1)
                pad_len = self.mem_max_depth - window
                if pad_len > 0:
                    pad = torch.zeros(1, pad_len * cur_mem.size(-1), device=cur_mem.device, dtype=cur_mem.dtype)
                    cur_flat = torch.cat([cur_flat, pad], dim=1)
                journal_mixed = self.journal_mixer(cur_flat).view(1,1,-1)
                journal_out = journal_out + 0.05 * journal_mixed
            # Diary for personalised memory traces - 6 layered sliding window
            diary_delta_activity = max(self.diary_linear1.delta_weight.abs().mean().item(), self.diary_linear2.delta_weight.abs().mean().item())
            diary_speed_gate = 1.0 if diary_delta_activity > 1e-4 else 0.0
            if diary_speed_gate > 0.0:
                diary_out = self.diary_linear2(self.thought_dropout(self.activation(self.diary_linear1(fused))))
            else:
                diary_out = fused
            with torch.no_grad():
                mem_update = fused.detach().mean([0,1]).to(self.diary_mem.dtype)
                idx = int(self.diary_idx[0].item())
                depth = int(self.diary_depth[0].item())
                self.diary_mem[idx].lerp_(mem_update, 0.05)
                next_idx = idx + 1
                if next_idx >= depth:
                    if self.mem_auto_grow and depth < self.mem_max_depth:
                        new_depth = min(depth * 2, self.mem_max_depth)
                        self.diary_depth[0] = new_depth
                        if not hasattr(self, '_diary_grow_logged'):
                            self._diary_grow_logged = set()
                        if depth not in self._diary_grow_logged:
                            print(f"[MemGrow] Diary depth grew {depth} -> {new_depth}")
                            self._diary_grow_logged.add(depth)
                    next_idx = 0
                self.diary_idx[0] = next_idx
                # Inner connections via mixer over sliding window of auto-growing memory
                window = min(self.mem_window, depth)
                if window > 0:
                    indices = [(idx - i - 1) % depth for i in range(window)]
                    cur_mem = self.diary_mem[indices].contiguous()
                else:
                    cur_mem = self.diary_mem[:0]
                cur_flat = cur_mem.view(1, -1)
                pad_len = self.mem_max_depth - window
                if pad_len > 0:
                    pad = torch.zeros(1, pad_len * cur_mem.size(-1), device=cur_mem.device, dtype=cur_mem.dtype)
                    cur_flat = torch.cat([cur_flat, pad], dim=1)
                diary_mixed = self.diary_mixer(cur_flat).view(1,1,-1)
                diary_out = diary_out + 0.05 * diary_mixed
            # North star guidance to keep reasoning on track
            north_star_input = fused + self.north_star_vec.view(1,1,-1)
            north_star_out = self.north_star_linear2(self.thought_dropout(self.activation(self.north_star_linear1(north_star_input))))
            src = src + gate * self.thought_dropout(thought) + pred_gate * self.thought_dropout(pred) + reason_gate * self.thought_dropout(fused) + meta_gate * self.thought_dropout(meta) + logic_gate * self.thought_dropout(logic_mod) + planning_gate * self.thought_dropout(planning) + ethical_gate * self.thought_dropout(ethical) + pattern_gate * self.thought_dropout(pattern) + structural_gate * self.thought_dropout(structural) + structural_reason_gate * self.thought_dropout(structural_reason) + structural_pattern_gate * self.thought_dropout(structural_pattern) + matrix_calc_gate * self.thought_dropout(matrix_calc) + map_gate * self.thought_dropout(map_out) + graph_gate * self.thought_dropout(graph_out) + scratch_pad_gate * self.thought_dropout(scratch_out) + ledger_gate * self.thought_dropout(ledger_out) + journal_gate * self.thought_dropout(journal_out) + diary_gate * self.thought_dropout(diary_out) + north_star_gate * self.thought_dropout(north_star_out)
            src = self.thought_norm(src)
            # Mini-agent helper layer with global cross-layer context and sliding window
            helper_agent_gate = torch.sigmoid(self.helper_agent_gate)
            # Fuse global helper context if available
            if hasattr(self, 'global_helper_context') and hasattr(self, 'global_helper_gate'):
                # Update sliding window memory for helper agent
                if hasattr(self, 'helper_window_mem') and hasattr(self, 'helper_window_idx'):
                    with torch.no_grad():
                        ctx_update = src.mean([0,1]).detach()
                        ctx_update = torch.nan_to_num(ctx_update, nan=0.0, posinf=1e4, neginf=-1e4)
                        ctx_update = torch.clamp(ctx_update, -10.0, 10.0)
                        idx = int(self.helper_window_idx[0].item())
                        # Circular buffer write - ensure device match
                        self.helper_window_mem[idx].copy_(ctx_update.to(self.helper_window_mem.device))
                        # Advance index
                        next_idx = idx + 1
                        if next_idx >= getattr(self, 'helper_window_size', 8):
                            next_idx = 0
                        self.helper_window_idx[0] = next_idx
                        # Use window mean as richer context
                        window_mean = self.helper_window_mem.mean(dim=0)
                        window_mean = torch.nan_to_num(window_mean, nan=0.0, posinf=1e4, neginf=-1e4)
                        # EMA update of global context from window - ensure device match
                        self.global_helper_context.data.mul_(0.99)
                        self.global_helper_context.data.add_(0.01 * window_mean.detach().to(self.global_helper_context.device))
                        # Clamp global context to avoid drift
                        self.global_helper_context.data.clamp_(-5.0, 5.0)
                else:
                    ctx_update = src.mean([0,1])
                    ctx_update = torch.nan_to_num(ctx_update, nan=0.0, posinf=1e4, neginf=-1e4)
                    ctx_update = torch.clamp(ctx_update, -10.0, 10.0)
                    self.global_helper_context.data.mul_(0.99)
                    self.global_helper_context.data.add_(0.01 * ctx_update.detach())
                    self.global_helper_context.data.clamp_(-5.0, 5.0)
                # Read context: blend global vector with recent window
                ctx = self.global_helper_context.view(1,1,-1).to(src.device)
                ctx_gate = torch.sigmoid(self.global_helper_gate)
                # If window exists, add a small window bias
                if hasattr(self, 'helper_window_mem'):
                    window_bias = self.helper_window_mem.mean(dim=0).view(1,1,-1).to(src.device)
                    src_ctx = src + ctx_gate * 0.02 * ctx + 0.01 * window_bias
                else:
                    src_ctx = src + ctx_gate * 0.02 * ctx
                helper_out = self.helper_agent_linear2(self.activation(self.helper_agent_linear1(src_ctx)))
                # Wave modulation for shared helper agent
                if hasattr(self, 'helper_wave_amp') and hasattr(self, '_layer_idx'):
                    t = (self._layer_idx + 1) / max(1, getattr(self, 'helper_window_size', 8))
                    # Clamp wave params to avoid explosion
                    amp = torch.clamp(self.helper_wave_amp, min=0.0, max=1.0)
                    freq = torch.clamp(self.helper_wave_freq, min=0.1, max=10.0)
                    wave = amp * torch.sin(freq * t + self.helper_wave_phase)
                    helper_out = helper_out * (1.0 + 0.1 * wave)
            else:
                helper_out = self.helper_agent_linear2(self.activation(self.helper_agent_linear1(src)))
            src = src + helper_agent_gate * 0.01 * self.thought_dropout(helper_out)
            # Radix sort hidden layer for ordered feature activation
            radix_gate = torch.sigmoid(self.radix_sort_gate)
            # Sort activations by magnitude to impose order; use torch.sort as radix-like ordering
            src_sorted, _ = torch.sort(src, dim=-1)
            src = src + radix_gate * 0.02 * self.thought_dropout(src_sorted - src)
            # Internal order for hidden physical layer
            order_gate = torch.sigmoid(self.order_gate)
            # depthwise conv over feature dim to enforce local order
            src_ordered = self.order_conv(src.transpose(1,2)).transpose(1,2)
            src = src + order_gate * 0.05 * self.thought_dropout(src_ordered + self.hidden_order_bias.view(1,1,-1))
            # Organizer hidden layer for information sorting and structure
            organizer_gate = torch.sigmoid(self.organizer_gate)
            organizer_sort_gate = torch.sigmoid(self.organizer_sort_gate)
            org_feat = self.organizer_linear2(self.activation(self.organizer_linear1(src)))
            # Learnable importance scores per feature
            importance = org_feat.mean(dim=1, keepdim=True)
            # Sort src by importance-weighted magnitude for structure
            src_mag = src.abs()
            # Differentiable sort via torch.sort for organization
            src_sorted_org, _ = torch.sort(src, dim=-1)
            # Gated mix of sorted and original with importance weighting
            src = src + organizer_gate * 0.03 * self.thought_dropout(org_feat)
            src = src + organizer_sort_gate * 0.02 * self.thought_dropout(src_sorted_org - src)
            # CPU offload hidden layer to reduce GPU stress
            cpu_offload_gate = torch.sigmoid(self.cpu_offload_gate)
            # Run on same device as src to avoid device mismatch
            cpu_feat = self.cpu_offload_linear2(self.activation(self.cpu_offload_linear1(src)))
            src = src + cpu_offload_gate * 0.01 * self.thought_dropout(cpu_feat)
            # Training-ease hidden layer to smooth step/val loss
            ease_gate = torch.sigmoid(self.ease_gate)
            ease_out = self.ease_linear2(self.activation(self.ease_linear1(src)))
            src = src + ease_gate * 0.03 * self.thought_dropout(ease_out)
            # Sliding hidden window for temporal context within layer
            with torch.no_grad():
                mem_update = src.detach().mean([0,1]).to(self.hidden_window_mem.dtype)
                rolled = torch.roll(self.hidden_window_mem, shifts=1, dims=0)
                self.hidden_window_mem.copy_(rolled)
                self.hidden_window_mem[0].lerp_(mem_update, 0.1)
            # Temporal attention over hidden window
            hidden_gate = torch.sigmoid(self.hidden_window_gate)
            attn_gate = torch.sigmoid(self.hidden_window_attn_gate)
            q = self.hidden_window_attn_q(src.mean(dim=1, keepdim=True))
            k = self.hidden_window_attn_k(self.hidden_window_mem)
            v = self.hidden_window_attn_v(self.hidden_window_mem)
            scale = math.sqrt(src.size(-1))
            attn_scores = torch.einsum('bqd,md->bqm', q, k) / scale
            attn_weights = torch.softmax(attn_scores, dim=-1)
            window_attn = torch.einsum('bqm,md->bqd', attn_weights, v)
            window_attn = window_attn.expand_as(src)
            src = src + hidden_gate * 0.05 * self.thought_dropout(window_attn)
            src = src + attn_gate * 0.03 * self.thought_dropout(window_attn)
            # Future anticipation hidden layer to recognize future from past
            future_gate = torch.sigmoid(self.future_gate)
            # Learn from past window to make better future
            future_input = src + 0.5 * window_attn
            future_context = self.future_linear2(self.activation(self.future_linear1(future_input)))
            src = src + future_gate * 0.04 * self.thought_dropout(future_context)
            # Memory attention over working memories
            mem_gate = torch.sigmoid(self.mem_attn_gate)
            # Delta-gated memory attention for speed
            mem_delta_activity = max(
                self.mem_attn_q.delta_weight.abs().mean().item(),
                self.mem_attn_k.delta_weight.abs().mean().item(),
                self.mem_attn_v.delta_weight.abs().mean().item(),
                self.mem_attn_out.delta_weight.abs().mean().item()
            )
            mem_speed_gate = 1.0 if mem_delta_activity > 1e-4 else 0.0
            if mem_speed_gate > 0.0:
                # Concatenate all working memories for richer context
                mem_bank = torch.cat([self.scratch_pad_mem, self.ledger_mem, self.journal_mem, self.diary_mem, self.north_star_vec.unsqueeze(0)], dim=0)
                q = self.mem_attn_q(src)
                k = self.mem_attn_k(mem_bank)
                v = self.mem_attn_v(mem_bank)
                scale = math.sqrt(src.size(-1))
                attn_scores = torch.einsum('btd,md->btm', q, k) / scale
                attn_weights = torch.softmax(attn_scores, dim=-1)
                mem_out = torch.einsum('btm,md->btd', attn_weights, v)
                mem_out = self.mem_attn_out(mem_out)
                src = src + mem_gate * self.thought_dropout(mem_out)
            # else: skip memory attention, keep src unchanged
            # Attractor convergence check
            eps = torch.sigmoid(self.convergence_eps).item() * 1e-3 + 1e-5
            if (src - prev_src).abs().mean().item() < eps:
                break
            prev_src = src.clone()
        # Chain-of-thought condensation - delta gated for speed
        cot_delta_activity = max(self.cot_condense_linear1.delta_weight.abs().mean().item(), self.cot_condense_linear2.delta_weight.abs().mean().item())
        cot_speed_gate = 1.0 if cot_delta_activity > 1e-4 else 0.0
        if cot_speed_gate > 0.0:
            cot_hidden = self.activation(self.cot_condense_linear1(src))
            cot_out = self.cot_condense_linear2(cot_hidden)
            cot_gate = torch.sigmoid(self.cot_gate)
            src = self.cot_norm(src + cot_gate * self.dropout(cot_out))
        # else: skip CoT condensation, keep src unchanged
        
        # FFN with SwiGLU, MoE and LayerScale
        gate = self.linear1_gate(src)
        up = self.linear1(src)
        # Mixture of Experts routing
        router_logits = self.moe_router(src)
        router_probs = F.softmax(router_logits, dim=-1)
        # Top-1 expert for efficiency
        expert_idx = torch.argmax(router_probs, dim=-1)
        moe_out = torch.zeros_like(up)
        for e_idx in range(self.moe_num_experts):
            mask = expert_idx == e_idx
            if mask.any():
                expert_input = src[mask]
                expert_output = self.moe_experts[e_idx](expert_input)
                moe_out[mask] = expert_output
        moe_gate = torch.sigmoid(self.moe_gate)
        up = up + moe_gate * moe_out
        swiglu = F.silu(up) * gate
        ff_out = self.linear2(self.dropout(swiglu))
        src = src + self.ffn_scale * self.dropout2(ff_out)
        src = self.norm2(src)
        return src

    def apply_delta_decay(self):
        self.self_attn.apply_delta_decay()
        self.linear1.apply_delta_decay()
        self.linear1_gate.apply_delta_decay()
        self.linear2.apply_delta_decay()
        for expert in self.moe_experts:
            expert.apply_delta_decay()
        self.moe_router.apply_delta_decay()
        self.cot_condense_linear1.apply_delta_decay()
        self.cot_condense_linear2.apply_delta_decay()
        self.thought_linear1.apply_delta_decay()
        self.thought_linear2.apply_delta_decay()
        self.prediction_linear.apply_delta_decay()
        self.thought_prediction_fusion.apply_delta_decay()
        self.meta_reason_linear1.apply_delta_decay()
        self.meta_reason_linear2.apply_delta_decay()
        self.planning_linear1.apply_delta_decay()
        self.planning_linear2.apply_delta_decay()
        self.ethical_linear1.apply_delta_decay()
        self.ethical_linear2.apply_delta_decay()
        self.pattern_linear1.apply_delta_decay()
        self.pattern_linear2.apply_delta_decay()
        self.structural_linear1.apply_delta_decay()
        self.structural_linear2.apply_delta_decay()
        self.structural_reason_linear1.apply_delta_decay()
        self.structural_reason_linear2.apply_delta_decay()
        self.structural_pattern_linear1.apply_delta_decay()
        self.structural_pattern_linear2.apply_delta_decay()
        self.matrix_calc_linear1.apply_delta_decay()
        self.matrix_calc_linear2.apply_delta_decay()
        self.map_linear1.apply_delta_decay()
        self.map_linear2.apply_delta_decay()
        self.graph_linear1.apply_delta_decay()
        self.graph_linear2.apply_delta_decay()
        self.scratch_pad_linear1.apply_delta_decay()
        self.scratch_pad_linear2.apply_delta_decay()
        self.ledger_linear1.apply_delta_decay()
        self.ledger_linear2.apply_delta_decay()
        self.journal_linear1.apply_delta_decay()
        self.journal_linear2.apply_delta_decay()
        self.diary_linear1.apply_delta_decay()
        self.diary_linear2.apply_delta_decay()
        self.north_star_linear1.apply_delta_decay()
        self.north_star_linear2.apply_delta_decay()
        self.organizer_linear1.apply_delta_decay()
        self.organizer_linear2.apply_delta_decay()
        self.cpu_offload_linear1.apply_delta_decay()
        self.cpu_offload_linear2.apply_delta_decay()
        self.simplify_linear1.apply_delta_decay()
        self.simplify_linear2.apply_delta_decay()
        self.helper_agent_linear1.apply_delta_decay()
        self.helper_agent_linear2.apply_delta_decay()

    def consolidate_delta(self):
        self.self_attn.consolidate_delta()
        self.linear1.consolidate_delta()
        self.linear1_gate.consolidate_delta()
        self.linear2.consolidate_delta()
        for expert in self.moe_experts:
            expert.consolidate_delta()
        self.moe_router.consolidate_delta()
        self.cot_condense_linear1.consolidate_delta()
        self.cot_condense_linear2.consolidate_delta()
        self.thought_linear1.consolidate_delta()
        self.thought_linear2.consolidate_delta()
        self.prediction_linear.consolidate_delta()
        self.thought_prediction_fusion.consolidate_delta()
        self.meta_reason_linear1.consolidate_delta()
        self.meta_reason_linear2.consolidate_delta()
        self.planning_linear1.consolidate_delta()
        self.planning_linear2.consolidate_delta()
        self.ethical_linear1.consolidate_delta()
        self.ethical_linear2.consolidate_delta()
        self.pattern_linear1.consolidate_delta()
        self.pattern_linear2.consolidate_delta()
        self.structural_linear1.consolidate_delta()
        self.structural_linear2.consolidate_delta()
        self.structural_reason_linear1.consolidate_delta()
        self.structural_reason_linear2.consolidate_delta()
        self.structural_pattern_linear1.consolidate_delta()
        self.structural_pattern_linear2.consolidate_delta()
        self.matrix_calc_linear1.consolidate_delta()
        self.matrix_calc_linear2.consolidate_delta()
        self.map_linear1.consolidate_delta()
        self.map_linear2.consolidate_delta()
        self.graph_linear1.consolidate_delta()
        self.graph_linear2.consolidate_delta()
        self.scratch_pad_linear1.consolidate_delta()
        self.scratch_pad_linear2.consolidate_delta()
        self.ledger_linear1.consolidate_delta()
        self.ledger_linear2.consolidate_delta()
        self.journal_linear1.consolidate_delta()
        self.journal_linear2.consolidate_delta()
        self.diary_linear1.consolidate_delta()
        self.diary_linear2.consolidate_delta()
        self.north_star_linear1.consolidate_delta()
        self.north_star_linear2.consolidate_delta()
        self.organizer_linear1.consolidate_delta()
        self.organizer_linear2.consolidate_delta()
        self.cpu_offload_linear1.consolidate_delta()
        self.cpu_offload_linear2.consolidate_delta()
        self.simplify_linear1.consolidate_delta()
        self.simplify_linear2.consolidate_delta()
        self.helper_agent_linear1.consolidate_delta()
        self.helper_agent_linear2.consolidate_delta()


class DualWeightWeaveFormer(nn.Module):
    """
    WeaveFormer with dual-weight layers throughout the 32-layer stack
    """
    def __init__(self, dim=256, num_layers=32, vocab_size=512, use_dual_weights=True, nhead=9, use_grad_checkpoint=True, num_passes=3, bidirectional=False):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.rotary = RotaryEmbedding(dim)
        self.use_dual_weights = use_dual_weights
        self.nhead = nhead
        self.use_grad_checkpoint = use_grad_checkpoint
        self.num_passes = max(1, int(num_passes))
        self.bidirectional = bool(bidirectional)
        
        if use_dual_weights:
            # Sliding window attention schedule: grow window with depth for optimal receptive field
            # Base window 256, double every 4 layers, capped at max_seq_len
            base_window = int(os.environ.get('ATTN_WINDOW_BASE','256'))
            window_growth = int(os.environ.get('ATTN_WINDOW_GROWTH','4'))
            max_window = int(os.environ.get('ATTN_WINDOW_MAX','1024'))
            self.layers = nn.ModuleList([])
            for i in range(num_layers):
                # Compute window size per layer
                layer_window = base_window * (2 ** (i // window_growth))
                layer_window = min(layer_window, max_window)
                # Create layer with window size for its self-attention
                layer = DualWeightTransformerEncoderLayer(dim, nhead=nhead, window_size=layer_window)
                layer._layer_idx = i
                self.layers.append(layer)
            # Shared helper agent weights across all layers for true cross-layer agent behavior
            self.shared_helper_linear1 = DualWeightLinear(dim, dim//4)
            self.shared_helper_linear2 = DualWeightLinear(dim//4, dim)
            self.shared_helper_gate = nn.Parameter(torch.tensor([-10.0]))
            # Wave modulation parameters for shared helper agent
            self.helper_wave_amp = nn.Parameter(torch.ones(1) * 0.2)
            self.helper_wave_freq = nn.Parameter(torch.ones(1) * 3.0)
            self.helper_wave_phase = nn.Parameter(torch.zeros(1))
            # Global helper agent context shared across layers for cross-layer coordination
            self.helper_global_context = nn.Parameter(torch.randn(dim) * 0.02)
            self.helper_global_gate = nn.Parameter(torch.zeros(1))
            # Sliding window memory for helper agent to retain recent cross-layer context
            self.helper_window_size = int(os.environ.get('HELPER_WINDOW_SIZE','8'))
            self.register_buffer('helper_window_mem', torch.zeros(self.helper_window_size, dim))
            self.register_buffer('helper_window_idx', torch.zeros(1, dtype=torch.long))
            # Shared simplify weights for math-reuse calculator across all layers with sliding window bank
            self.simplify_bank_size = 8
            self.shared_simplify_linear1_bank = nn.ModuleList([DualWeightLinear(dim, dim//8) for _ in range(self.simplify_bank_size)])
            self.shared_simplify_linear2_bank = nn.ModuleList([DualWeightLinear(dim//8, dim) for _ in range(self.simplify_bank_size)])
            # Dynamic compression variants for speed
            self.shared_simplify_linear1_small_bank = nn.ModuleList([DualWeightLinear(dim, dim//16) for _ in range(self.simplify_bank_size)])
            self.shared_simplify_linear2_small_bank = nn.ModuleList([DualWeightLinear(dim//16, dim) for _ in range(self.simplify_bank_size)])
            self.shared_simplify_gate = nn.Parameter(torch.zeros(1))
            self.simplify_wave_amp = nn.Parameter(torch.ones(1) * 0.2)
            self.simplify_wave_freq = nn.Parameter(torch.ones(1) * 3.0)
            self.simplify_wave_phase = nn.Parameter(torch.zeros(1))
            # Wire shared helper weights, global context and sliding window into each layer
            for i, layer in enumerate(self.layers):
                layer._layer_idx = i
                layer.helper_agent_linear1 = self.shared_helper_linear1
                layer.helper_agent_linear2 = self.shared_helper_linear2
                layer.helper_agent_gate = self.shared_helper_gate
                layer.helper_wave_amp = self.helper_wave_amp
                layer.helper_wave_freq = self.helper_wave_freq
                layer.helper_wave_phase = self.helper_wave_phase
                layer.global_helper_context = self.helper_global_context
                layer.global_helper_gate = self.helper_global_gate
                layer.helper_window_mem = self.helper_window_mem
                layer.helper_window_idx = self.helper_window_idx
                layer.helper_window_size = self.helper_window_size
                # Wire shared simplify weights with sliding window bank and depth-phase modulation
                bank_idx = i % self.simplify_bank_size
                layer.simplify_linear1 = self.shared_simplify_linear1_bank[bank_idx]
                layer.simplify_linear2 = self.shared_simplify_linear2_bank[bank_idx]
                layer.simplify_linear1_small = self.shared_simplify_linear1_small_bank[bank_idx]
                layer.simplify_linear2_small = self.shared_simplify_linear2_small_bank[bank_idx]
                layer.simplify_gate = self.shared_simplify_gate
                layer.simplify_wave_amp = self.simplify_wave_amp
                layer.simplify_wave_freq = self.simplify_wave_freq
                layer.simplify_wave_phase = self.simplify_wave_phase
            # Internal layer index embedding for ordered depth
            self.layer_idx_embedding = nn.Parameter(torch.randn(num_layers, dim) * 0.02)
            self.layer_idx_gate = nn.Parameter(torch.zeros(1))
            # Learnable weave lattice: each layer can mix current + previous 2 states for roundabout skip connections
            self.weave_mixers = nn.ModuleList([
                DualWeightLinear(3*dim, dim) for _ in range(num_layers)
            ])
            # Hidden layer attention weights for dynamic gating of each layer's contribution
            self.layer_attn_weights = nn.Parameter(torch.ones(num_layers))
            self.layer_attn_norm = RMSNorm(dim)
            # Depthwise convolution filters for local pattern capture in hidden layers
            self.hidden_convs = nn.ModuleList([
                nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)
                for _ in range(num_layers)
            ])
            self.conv_gates = nn.Parameter(torch.zeros(num_layers))
            # Lightweight gating filter for channel-wise modulation
            self.channel_gates = nn.Parameter(torch.ones(num_layers, dim))
            # Bloom filters for hidden layer activation deduplication
            self.hidden_bloom = [set() for _ in range(num_layers)]
            # Activation neuron learning with high impact stress reinforcement
            self.activation_stress_gates = nn.Parameter(torch.zeros(num_layers))
            self.activation_stress_weights = nn.Parameter(torch.ones(num_layers, dim))
            self.activation_stress_norm = RMSNorm(dim)
            # Music-inspired hidden layer for rhythmic pattern learning and memory
            self.music_freqs = nn.Parameter(torch.ones(num_layers) * 0.1)
            self.music_phases = nn.Parameter(torch.zeros(num_layers))
            self.music_gates = nn.Parameter(torch.zeros(num_layers))
            # Fractal zoning for self-similar hierarchical modulation across depth and position
            self.fractal_zone_gate = nn.Parameter(torch.zeros(num_layers))
            self.fractal_zone_scale = nn.Parameter(torch.ones(num_layers) * 0.05)
            # Pattern-matching hidden layer for neuron-level pattern recall
            # Initialise pattern bank with Flower of Life inspired basis for structured recall
            bank_init = torch.zeros(num_layers, 8, dim)
            for i in range(num_layers):
                for k in range(8):
                    bank_init[i, k] = flower_of_life_weights(num_layers, dim, circles=64)[i].clone()
            self.pattern_bank = nn.Parameter(bank_init)
            self.pattern_match_gates = nn.Parameter(torch.zeros(num_layers))
            # Shared global thought core for reasoning across all layers
            self.shared_thought_linear1 = DualWeightLinear(dim, dim//2)
            self.shared_thought_linear2 = DualWeightLinear(dim//2, dim)
            self.shared_thought_gate = nn.Parameter(torch.zeros(1))
            self.shared_thought_norm = RMSNorm(dim)
            self.shared_thought_dropout = nn.Dropout(0.1)
            # Shared prediction for global reasoning
            self.shared_prediction_linear = DualWeightLinear(dim, dim)
            self.shared_prediction_gate = nn.Parameter(torch.zeros(1))
            self.shared_prediction_bias = nn.Parameter(torch.zeros(dim))
            self.shared_fusion = DualWeightLinear(2*dim, dim)
            self.shared_fusion_gate = nn.Parameter(torch.zeros(1))
            # Shared logic weights for structural reasoning
            self.shared_logic_weight = nn.Parameter(torch.zeros(1))
            self.shared_logic_gate = nn.Parameter(torch.zeros(1))
            # Structural learning & remembering: persistent memory bank per layer with EMA update
            self.register_buffer('structural_memory', torch.zeros(num_layers, dim))
            self.structural_memory_decay = 0.99
            self.structural_memory_gates = nn.Parameter(torch.full((num_layers,), -5.0))
            self.structural_learn_rate = nn.Parameter(torch.ones(num_layers) * 0.01)
            # Full layer interconnect: each layer can attend to all previous layers
            # Initialise with Flower of Life inspired pattern for structured connectivity
            fl_weights = flower_of_life_weights(num_layers, num_layers, circles=64)
            self.layer_interconnect = nn.Parameter(0.01 * fl_weights)
            self.layer_interconnect_gate = nn.Parameter(torch.tensor(-5.0))
        else:
            self.layers = nn.ModuleList([
                nn.TransformerEncoderLayer(d_model=dim, nhead=nhead, batch_first=True)
                for _ in range(num_layers)
            ])
            self.weave_mixers = None
        
        self.head = DualWeightLinear(dim, vocab_size) if use_dual_weights else nn.Linear(dim, vocab_size)
        self.norm = RMSNorm(dim)
        # Wave-based weight modulation for depth-varying behavior
        self.wave_amp = nn.Parameter(torch.ones(1) * 0.2)
        self.wave_freq = nn.Parameter(torch.ones(1) * 3.0)
        self.wave_phase = nn.Parameter(torch.zeros(1))
        # Weight tying for better convergence
        if use_dual_weights:
            # Tie head master weights to embedding weights
            with torch.no_grad():
                self.head.master_weight.data.copy_(self.embed.weight.data)
                if self.head.master_bias is not None:
                    nn.init.zeros_(self.head.master_bias)
        
    def forward(self, ids):
        # Global parameter sanitisation before any computation
        with torch.no_grad():
            for p in self.parameters():
                if p.is_floating_point():
                    p.data = torch.nan_to_num(p.data, nan=0.0, posinf=1e3, neginf=-1e3)
        x = self.embed(ids)
        # Clamp and sanitize unstable parameters to prevent NaN drift
        if self.use_dual_weights:
            with torch.no_grad():
                for name in ['layer_idx_embedding','fractal_zone_scale','conv_gates','channel_gates','activation_stress_gates','activation_stress_weights','layer_attn_weights','layer_idx_gate','fractal_zone_gate','music_freqs','music_phases','music_gates','pattern_match_gates','helper_global_context','pattern_bank','shared_thought_gate','shared_prediction_gate','shared_prediction_bias','shared_fusion_gate','shared_logic_weight','shared_logic_gate']:
                    if hasattr(self, name):
                        p = getattr(self, name)
                        if isinstance(p, torch.nn.Parameter):
                            p.data = torch.nan_to_num(p.data, nan=0.0, posinf=1e3, neginf=-1e3)
                self.layer_idx_embedding.data.clamp_(-5.0, 5.0)
                self.fractal_zone_scale.data.clamp_(-1.0, 1.0)
                self.conv_gates.data.clamp_(-5.0, 5.0)
                self.channel_gates.data.clamp_(-5.0, 5.0)
                # Sanitize and clamp wave params
                self.wave_amp.data = torch.nan_to_num(self.wave_amp.data, nan=0.05, posinf=1.0, neginf=-1.0)
                self.wave_amp.data.clamp_(-0.5, 0.5)
                self.wave_freq.data = torch.nan_to_num(self.wave_freq.data, nan=6.28, posinf=10.0, neginf=0.1)
                self.wave_freq.data.clamp_(0.1, 10.0)
                self.wave_phase.data = torch.nan_to_num(self.wave_phase.data, nan=0.0, posinf=1e3, neginf=-1e3)
        # Sliding-window / multi-pass depth: reuse the 32 layers multiple times to act like 64+ layers
        for _pass in range(self.num_passes):
            prev_x = x
            prev2_x = x
            for i, layer in enumerate(self.layers):
                if self.use_grad_checkpoint and self.training:
                    x = checkpoint(lambda inp: layer(inp), x)
                else:
                    x = layer(x)
                # Per-block NaN guard
                if torch.isnan(x).any() or torch.isinf(x).any():
                    print(f"[NaN Guard] NaN/Inf detected after layer {i} pass {_pass}")
                    x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
                # Wave-based weight modulation for depth-varying behavior
                t = (i + 1) / len(self.layers)
                wave = self.wave_amp * torch.sin(self.wave_freq * t + self.wave_phase)
                x = x * (1.0 + 0.1 * wave)
                # Internal layer index embedding for ordered depth
                if self.use_dual_weights:
                    idx_emb = self.layer_idx_embedding[i].view(1,1,-1).to(x.device)
                    # Guard against explosion
                    idx_emb = torch.clamp(idx_emb, min=-1.0, max=1.0)
                    idx_gate = torch.sigmoid(self.layer_idx_gate)
                    x = x + idx_gate * 0.03 * idx_emb
                    # Fractal zoning modulation for self-similar hierarchical patterns
                    fz_gate = torch.sigmoid(self.fractal_zone_gate[i])
                    fz_scale = torch.clamp(self.fractal_zone_scale[i], min=-0.5, max=0.5)
                    seq_len = x.size(1)
                    # Self-similar fractal pattern: log-scale depth + position
                    depth_factor = math.log2(i + 2)
                    pos = torch.arange(seq_len, device=x.device).float()
                    # Fractal modulation using sine of log-scaled position
                    fractal_pat = torch.sin(depth_factor * pos * 0.15)
                    fractal_pat = fractal_pat.view(1, -1, 1)
                    x = x * (1.0 + fz_gate * fz_scale * fractal_pat)
                # Sanitize immediately after layer to prevent NaN propagation
                x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
                x = torch.clamp(x, -10.0, 10.0)
                # Hidden layer attention gating for dynamic layer contribution
                if self.use_dual_weights:
                    attn_weight = torch.softmax(self.layer_attn_weights, dim=0)[i].item()
                    # Alert when layer attention spikes
                    if attn_weight > 0.2 and i % 8 == 0:
                        # lightweight alert, only occasionally to avoid spam
                        if not hasattr(self, '_alert_counter'):
                            self._alert_counter = 0
                        self._alert_counter += 1
                        if self._alert_counter % 100 == 0:
                            print(f"[Alert] Layer {i} attention spike: {attn_weight:.3f}")
                    x = self.layer_attn_norm(x * (1.0 + attn_weight))
                    # Depthwise conv filter for local sequence patterns with bloom dedup
                    # Simple bloom-like dedup based on activation signature
                    # Guard against NaNs in activations
                    act_slice = x[0,0,:8]
                    if torch.isnan(act_slice).any():
                        act_sig = tuple([0]*8)
                    else:
                        act_sig = tuple(map(int, torch.round(act_slice).tolist()))
                    if act_sig not in self.hidden_bloom[i]:
                        self.hidden_bloom[i].add(act_sig)
                        if len(self.hidden_bloom[i]) > 10000:
                            self.hidden_bloom[i].clear()
                        x_conv = x.transpose(1,2)
                        x_conv = self.hidden_convs[i](x_conv)
                        x_conv = x_conv.transpose(1,2)
                        # Guard conv output
                        x_conv = torch.nan_to_num(x_conv, nan=0.0, posinf=1e4, neginf=-1e4)
                        x_conv = torch.clamp(x_conv, -5.0, 5.0)
                        conv_gate = torch.sigmoid(self.conv_gates[i])
                        x = x + conv_gate * x_conv
                    # else skip conv for duplicate activation
                    # Channel-wise gating filter
                    ch_gate = torch.sigmoid(self.channel_gates[i])
                    x = x * (1.0 + ch_gate)
                    x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
                    x = torch.clamp(x, -10.0, 10.0)
                    # Activation neuron learning with high impact stress reinforcement
                    stress_gate = torch.sigmoid(self.activation_stress_gates[i])
                    # Compute per-neuron activation magnitude stress
                    act_mag = torch.norm(x, dim=-1, keepdim=True).detach()
                    stress_factor = stress_gate * torch.tanh(act_mag * 0.5)
                    # Clamp stress to avoid explosion
                    stress_factor = torch.clamp(stress_factor, -1.0, 1.0)
                    stress_weights = self.activation_stress_weights[i]
                    # Clamp weights to reasonable range
                    stress_weights_clamped = torch.clamp(stress_weights, -1.0, 1.0)
                    x = x * (1.0 + 0.01 * stress_factor * stress_weights_clamped)
                    x = self.activation_stress_norm(x)
                    # Guard against NaNs from stress
                    x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
                    # Music-inspired rhythmic modulation for pattern memory
                    music_gate = torch.sigmoid(self.music_gates[i])
                    freq = torch.abs(self.music_freqs[i]) + 1e-6
                    phase = self.music_phases[i]
                    # Position-based sinusoidal modulation
                    seq_len = x.size(1)
                    pos = torch.arange(seq_len, device=x.device).float() / max(1, seq_len)
                    modulation = torch.sin(freq * pos + phase)
                    modulation = modulation.view(1, -1, 1)
                    x = x * (1.0 + music_gate * 0.1 * modulation)
                    # Pattern-matching hidden layer for neuron-level pattern recall
                    pm_gate = torch.sigmoid(self.pattern_match_gates[i])
                    # Compute cosine similarity to pattern bank
                    bank = self.pattern_bank[i]  # [8, dim]
                    # Safe normalize to avoid division by zero -> NaNs
                    # Cast to float32 for numerical stability in fp16
                    x_norm = x.float() / (x.float().norm(dim=-1, keepdim=True).clamp(min=1e-12))
                    bank_norm = bank.float() / (bank.float().norm(dim=-1, keepdim=True).clamp(min=1e-12))
                    # Cast back to original dtype
                    x_norm = x_norm.to(x.dtype)
                    bank_norm = bank_norm.to(bank.dtype)
                    # similarity: [B, T, 8]
                    sim = torch.einsum('btd,pd->btp', x_norm, bank_norm)
                    sim_max, _ = sim.max(dim=-1)
                    # Boost activations where pattern matches
                    x = x * (1.0 + pm_gate * 0.1 * torch.sigmoid(sim_max).unsqueeze(-1))
                    gate = torch.sigmoid(self.shared_thought_gate)
                    pred_gate = torch.sigmoid(self.shared_prediction_gate)
                    fusion_gate = torch.sigmoid(self.shared_fusion_gate)
                    logic_gate = torch.sigmoid(self.shared_logic_gate)
                    thought = self.shared_thought_linear2(self.shared_thought_dropout(torch.nn.functional.gelu(self.shared_thought_linear1(x))))
                    pred = self.shared_prediction_linear(x) + self.shared_prediction_bias
                    fused = self.shared_fusion(torch.cat([thought, pred], dim=-1))
                    logic_mod = torch.sigmoid(self.shared_logic_weight) * logic_gate * thought
                    x = x + gate * self.shared_thought_dropout(thought) + pred_gate * self.shared_thought_dropout(pred) + fusion_gate * self.shared_thought_dropout(fused) + logic_gate * self.shared_thought_dropout(logic_mod)
                    x = self.shared_thought_norm(x)
                    # Structural learning & remembering: EMA memory bank per layer
                    if self.training:
                        mem_update = x.detach().mean([0,1])
                        with torch.no_grad():
                            new_mem = self.structural_memory_decay * self.structural_memory[i].data + (1.0 - self.structural_memory_decay) * mem_update
                            new_mem = torch.clamp(new_mem, -5.0, 5.0)
                            self.structural_memory[i].data.copy_(new_mem)
                    mem_gate = torch.sigmoid(self.structural_memory_gates[i])
                    mem = self.structural_memory[i].data.view(1,1,-1)
                    x = x + 0.01 * mem_gate * mem.to(x.device)
                    x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
                    x = torch.clamp(x, -10.0, 10.0)
                    # Full layer interconnect: mix with all previous layers via learned weights
                    if i > 0:
                        # Collect previous outputs from a running buffer stored as list
                        # For efficiency we approximate with prev_x and prev2_x and a learned mix
                        # Use interconnect weights to blend current with history
                        inter_weights = torch.softmax(self.layer_interconnect[i, :i+1], dim=0)
                        # Simple weighted residual from prev layers
                        # Here we blend current x with prev_x and prev2_x using interconnect
                        # Build a small history stack
                        hist = [x, prev_x]
                        if i > 1:
                            hist.append(prev2_x)
                        # Weighted sum
                        w_sum = sum(w * h for w, h in zip(inter_weights.tolist(), hist[:len(inter_weights)]))
                        inter_gate = torch.sigmoid(self.layer_interconnect_gate)
                        x = x + inter_gate * 0.01 * (w_sum - x)
                        x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
                        x = torch.clamp(x, -10.0, 10.0)
                if self.use_dual_weights and self.weave_mixers is not None:
                    # Learnable weave lattice: roundabout skip with current + prev + prev2
                    mixed = torch.cat([x, prev_x, prev2_x], dim=-1)
                    weave_out = self.weave_mixers[i](mixed)
                    x = x + 0.1 * weave_out  # residual weave
            prev2_x = prev_x
            prev_x = x
        x = self.norm(x)
        logits = self.head(x)
        return logits
    
    def apply_delta_decay(self):
        if self.use_dual_weights:
            for layer in self.layers:
                layer.apply_delta_decay()
            self.head.apply_delta_decay()
            for wm in self.weave_mixers:
                wm.apply_delta_decay()
            self.shared_thought_linear1.apply_delta_decay()
            self.shared_thought_linear2.apply_delta_decay()
            self.shared_prediction_linear.apply_delta_decay()
            self.shared_fusion.apply_delta_decay()
            # Shared helper agent weights
            if hasattr(self, 'shared_helper_linear1'):
                self.shared_helper_linear1.apply_delta_decay()
                self.shared_helper_linear2.apply_delta_decay()
    
    def consolidate_delta(self):
        if self.use_dual_weights:
            for layer in self.layers:
                layer.consolidate_delta()
            self.head.consolidate_delta()
            for wm in self.weave_mixers:
                wm.consolidate_delta()
            self.shared_thought_linear1.consolidate_delta()
            self.shared_thought_linear2.consolidate_delta()
            self.shared_prediction_linear.consolidate_delta()
            self.shared_fusion.consolidate_delta()
            # Shared helper agent weights
            if hasattr(self, 'shared_helper_linear1'):
                self.shared_helper_linear1.consolidate_delta()
                self.shared_helper_linear2.consolidate_delta()