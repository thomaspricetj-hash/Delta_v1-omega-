"""
WeaveFormer – neural predictor on pattern IDs
With PredictiveLinear wrapper for matrix-multiply predictor offload
"""
import torch
import torch.nn as nn
from pathlib import Path
import hashlib

# Import predictor utilities
try:
    from infrastructure.predictors.matrix_multiply_predictor import _CLUSTER_STATS, _magnitude_bin, _cluster_key
except Exception:
    _CLUSTER_STATS = {}
    def _magnitude_bin(v): return 0
    def _cluster_key(op,a,b): return (op,0,0)

class PredictiveLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.use_predictor = True  # predictor always on
        self.cache = {}
        self.hit_count = 0
        self.miss_count = 0

    def _signature(self, x):
        # Include shape in signature to avoid collisions across different sequence lengths
        if x.numel() == 0:
            return b''
        flat = x.detach().flatten()[:8]
        shape_bytes = str(x.shape).encode()
        h = hashlib.sha256(shape_bytes + flat.cpu().numpy().tobytes()).hexdigest()[:16]
        return h

    def forward(self, x):
        if not self.use_predictor:
            return self.linear(x)
        sig = self._signature(x)
        if sig in self.cache:
            self.hit_count += 1
            return self.cache[sig]
        out = self.linear(x)
        self.cache[sig] = out
        self.miss_count += 1
        return out

    def hit_rate(self):
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0.0

    def enable(self, flag=True):
        self.use_predictor = flag

class WeaveFormer(nn.Module):
    def __init__(self, dim=256, num_layers=2, vocab_size=512, nhead=9):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.nhead = nhead
        self.layers = nn.ModuleList([self._make_encoder_layer(dim, nhead) for _ in range(num_layers)])
        self.head = PredictiveLinear(dim, vocab_size)
        # Weave Lattice with bridges for inter-layer mixing
        try:
            from infrastructure.weave.weave_lattice import WeaveLatticeWrapper
            self.weave_lattice = WeaveLatticeWrapper(dim=dim, num_layers=num_layers, bridge_density=0.25)
            self.lattice_enabled = True
        except Exception:
            self.weave_lattice = None
            self.lattice_enabled = False
        self.predictor_mode = True
        self.use_checkpoint = False
        self.noise_scale = 0.01  # adaptive noise for regularization
        # Self-awareness head for in-model introspection
        try:
            from .awareness import AwarenessHead
            self.awareness_head = AwarenessHead(dim=dim)
            self.awareness_enabled = True
        except Exception:
            self.awareness_head = None
            self.awareness_enabled = False
        # Math predictor for inference speed on math spans
        try:
            from infrastructure.predictors.math_predictor import MathPredictor
            self.math_predictor = MathPredictor()
        except Exception:
            self.math_predictor = None
        # MBFN bridge for explainable ranking
        try:
            from infrastructure.mbfn.store import FrameStore
            from infrastructure.mbfn.bridge import MBFNBridge
            self.mbfn_store = FrameStore()
            self.mbfn_bridge = MBFNBridge(self.mbfn_store, alpha=0.7)
        except Exception:
            self.mbfn_store = None
            self.mbfn_bridge = None
        # MDWE sparse cognitive routing projector - differentiable wrapper with heat gating
        try:
            from infrastructure.mdwe_engine.heat_gated_wrapper import HeatGatedMDWEProjector
            # Heatmap will be injected later if available
            self.mdwe_projector = HeatGatedMDWEProjector(num_neurons=1024, dim=dim, heatmap=None)
            self.mdwe_enabled = True
            self._mdwe_heat_gated = True
        except Exception:
            try:
                from infrastructure.mdwe_engine.torch_wrapper import MDWESparseProjector
                self.mdwe_projector = MDWESparseProjector(num_neurons=1024, dim=dim)
                self.mdwe_enabled = True
                self._mdwe_heat_gated = False
            except Exception:
                self.mdwe_projector = None
                self.mdwe_enabled = False
                self._mdwe_heat_gated = False

    def _make_encoder_layer(self, dim, nhead):
        # nhead must divide dim
        return nn.TransformerEncoderLayer(d_model=dim, nhead=nhead, batch_first=True)

    def forward(self, ids: torch.Tensor):
        x = self.embed(ids)
        # Adaptive noise injection for regularization and exploration
        if self.training and self.noise_scale > 0:
            noise = torch.randn_like(x) * self.noise_scale
            x = x + noise
        # Lattice inner connections: accumulate all layer outputs for richer gradients
        xs = [x]
        layer_outputs = []
        for layer in self.layers:
            if self.use_checkpoint:
                from torch.utils.checkpoint import checkpoint
                x = checkpoint(layer, x)
            else:
                x = layer(x)
            layer_outputs.append(x)
            xs.append(x)
        # Weave Lattice with bridges for inter-layer mixing
        if self.lattice_enabled and self.weave_lattice is not None:
            x = self.weave_lattice(x, layer_outputs)
        else:
            # Fallback mean pooling
            x = sum(xs) / len(xs)
        # Self-awareness gating
        awareness_gates = None
        if self.awareness_enabled and self.awareness_head is not None:
            awareness_gates, _ = self.awareness_head(x)
            # Modulate MDWE gate strength by confidence gate
            if self.mdwe_enabled and self.mdwe_projector is not None:
                # Inject gates into projector if it supports it
                try:
                    if hasattr(self.mdwe_projector, 'set_awareness_gates'):
                        self.mdwe_projector.set_awareness_gates(awareness_gates)
                except Exception:
                    pass
        # MDWE sparse cognitive routing projection - differentiable via straight-through estimator
        if self.mdwe_enabled and self.mdwe_projector is not None:
            x = self.mdwe_projector(x)
        logits = self.head(x)
        # Return logits and awareness gates for introspection
        if awareness_gates is not None:
            return logits, awareness_gates
        return logits

    def infer_with_math(self, ids: torch.Tensor, tokenizer=None):
        """Inference with math predictor routing for speed"""
        if self.math_predictor is None:
            return self.forward(ids)
        # If tokenizer provided and batch size 1, try math shortcut
        if tokenizer is not None and ids.shape[0] == 1:
            try:
                token_list = ids[0].tolist()
                token_list = [t for t in token_list if t != 0]
                text = tokenizer.decode(token_list)
                if self.math_predictor.is_math(text):
                    expr = text.strip()
                    result = self.math_predictor.evaluate(expr)
                    if result is not None:
                        result_ids = tokenizer.encode(result)
                        import torch
                        out = torch.zeros_like(ids)
                        out[0, :len(result_ids)] = torch.tensor(result_ids, dtype=ids.dtype)
                        return out
            except Exception:
                pass
        return self.forward(ids)

    def generate_with_math_shortcut(self, prompt: str, tokenizer, max_new_tokens: int = 64):
        """Generate with math shortcut for token generation speed"""
        if self.math_predictor is None:
            raise RuntimeError("MathPredictor not available")
        # Check if prompt itself is math
        if self.math_predictor.is_math(prompt):
            result = self.math_predictor.evaluate(prompt)
            if result:
                return result
        # Fallback to normal generation with optional MBFN re-ranking
        return self.generate_with_mbfn(prompt, tokenizer, max_new_tokens)

    def generate_with_mbfn(self, prompt: str, tokenizer, max_new_tokens: int = 64, top_k: int = 8):
        """Generate with MBFN evidence re-ranking for speed and quality"""
        try:
            from infrastructure.hscl.hscl_preprocessor import compress as hscl_compress
        except Exception:
            def hscl_compress(t): return t
        comp_prompt = hscl_compress(prompt)
        ids = tokenizer.encode(comp_prompt)
        input_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
        # Activate MBFN frames from prompt
        if self.mbfn_bridge is not None:
            self.mbfn_bridge.store.activate_frames([])
            # simple keyword activation
            words = set(comp_prompt.lower().split())
            triggers = [(w, 1.0) for w in words if len(w) > 3]
            self.mbfn_bridge.store.activate_frames(triggers)
        output_ids = list(ids)
        for _ in range(max_new_tokens):
            with torch.no_grad():
                logits = self.forward(input_ids)
                last_logits = logits[0, -1]
                top_vals, top_idx = torch.topk(last_logits, k=top_k)
                candidates = [tokenizer.decode([int(i)]) for i in top_idx.tolist()]
                scores = top_vals.tolist()
                if self.mbfn_bridge is not None:
                    ranked = self.mbfn_bridge.rank_candidates(candidates, scores)
                    chosen_text = ranked[0][0]
                else:
                    chosen_text = candidates[0]
                chosen_ids = tokenizer.encode(chosen_text)
                if not chosen_ids:
                    break
                output_ids.extend(chosen_ids)
                input_ids = torch.tensor(output_ids, dtype=torch.long).unsqueeze(0)
        return tokenizer.decode(output_ids)

    def set_heatmap(self, heatmap):
        if getattr(self, '_mdwe_heat_gated', False) and self.mdwe_projector is not None:
            try:
                self.mdwe_projector.heatmap = heatmap
            except Exception:
                pass
        return self

    def enable_predictor_mode(self, threshold=0.9):
        self.predictor_mode = True
        for name, module in self.named_modules():
            if isinstance(module, PredictiveLinear):
                module.enable(True)
        return self

    def disable_predictor_mode(self):
        self.predictor_mode = False
        for name, module in self.named_modules():
            if isinstance(module, PredictiveLinear):
                module.enable(False)
        return self

    def predictor_stats(self):
        stats = {}
        for name, module in self.named_modules():
            if isinstance(module, PredictiveLinear):
                stats[name] = {
                    'hits': module.hit_count,
                    'misses': module.miss_count,
                    'hit_rate': module.hit_rate()
                }
        return stats
