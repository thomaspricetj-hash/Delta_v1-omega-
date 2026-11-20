"""
Math-aware WeaveFormer - best route wiring
Matrix multiply predictor -> embeddings / attention / FFN / next-token probs
"""
import re
import json
import torch
import torch.nn as nn
from .weaveformer import WeaveFormer
import sys
from pathlib import Path
BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE))
from infrastructure.predictors.matrix_multiply_predictor import cached_matmul
from infrastructure.tokenizer.bpe_tokenizer import get_bpe_tokenizer

MATH_RE = re.compile(r'[-+]?\d+\.?\d*\s*([\*\+\-\/])\s*[-+]?\d+\.?\d*')

class MathAwareWeaveFormer(nn.Module):
    def __init__(self, dim=256, num_layers=2, vocab_size=512, math_blend=0.3):
        super().__init__()
        self.base = WeaveFormer(dim, num_layers, vocab_size)
        self.math_blend = math_blend
        self.tok = get_bpe_tokenizer()
        self.vocab_size = vocab_size
        # Learnable gate for math injection
        self.gate = nn.Parameter(torch.tensor(0.1))
        self._adaptive_blend = True
        self._blend_min = 0.2
        self._blend_max = 0.9
        # Per-op blend cache and EMA cache for efficiency
        self._ema_cache = {}
        self._ema_mtime = 0
        self._per_op_blend = {op: math_blend for op in ['+','-','*','/']}
        
    def _parse_math(self, text):
        m = MATH_RE.search(text)
        if not m:
            return None
        expr = m.group(0)
        op = m.group(1)
        parts = re.split(r'[\*\+\-\/]', expr)
        if len(parts) != 2:
            return None
        try:
            a = float(parts[0].strip())
            b = float(parts[1].strip())
            res = cached_matmul(a, b, op)
            if res is None:
                return None
            return float(res), op, a, b
        except Exception:
            return None

    def _result_token_id(self, value):
        try:
            txt = str(int(value)) if float(value).is_integer() else str(value)
            toks = self.tok.encode(txt)
            if toks:
                return int(min(toks[0], self.vocab_size-1))
        except Exception:
            pass
        return None

    def _load_ema(self):
        # Load latest EMA per op with mtime cache
        try:
            ema_path = BASE / "runtime" / "logs" / "predictor_ema.jsonl"
            if not ema_path.exists():
                return self._ema_cache
            mtime = ema_path.stat().st_mtime
            if mtime <= self._ema_mtime and self._ema_cache:
                return self._ema_cache
            latest = {}
            with open(ema_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        op = obj.get('op')
                        if op:
                            latest[op] = float(obj.get('ema', 0.0))
                    except Exception:
                        continue
            self._ema_cache = latest
            self._ema_mtime = mtime
            return latest
        except Exception:
            return self._ema_cache

    def _adaptive_math_blend(self):
        # Adjust math_blend based on predictor EMA and confidence, per-op
        try:
            from infrastructure.predictors.matrix_multiply_predictor import _PREDICTOR_STATS
            ema_map = self._load_ema()
            # Update per-op blends
            for op in ['+','-','*','/']:
                if ema_map and op in ema_map:
                    ema = ema_map[op]
                    # Map EMA 0.5-1.0 -> blend 0.2-0.9
                    target = self._blend_min + (ema - 0.5) / (1.0 - 0.5) * (self._blend_max - self._blend_min)
                else:
                    conf = _PREDICTOR_STATS.get(op,{}).get('confidence',0.85)
                    target = self._blend_min + (conf-0.85)/(0.99-0.85)*(self._blend_max-self._blend_min)
                target = max(self._blend_min, min(self._blend_max, target))
                # Smooth per-op
                prev = self._per_op_blend.get(op, self.math_blend)
                self._per_op_blend[op] = 0.9*prev + 0.1*target
            # Global blend as average of per-op
            avg_blend = sum(self._per_op_blend.values())/len(self._per_op_blend)
            self.math_blend = 0.9*self.math_blend + 0.1*avg_blend
            # Export Prometheus gauge
            self._export_blend_metrics()
        except Exception:
            pass

    def _export_blend_metrics(self):
        try:
            prom_path = BASE / "runtime" / "logs" / "math_blend.prom"
            with open(prom_path,'w',encoding='utf-8') as f:
                f.write('# HELP math_blend_current Current math blend value\n')
                f.write('# TYPE math_blend_current gauge\n')
                f.write(f'math_blend_current {self.math_blend:.6f}\n')
                f.write('# HELP math_blend_per_op Per-op math blend\n')
                f.write('# TYPE math_blend_per_op gauge\n')
                for op, val in self._per_op_blend.items():
                    f.write(f'math_blend_per_op{{op="{op}"}} {val:.6f}\n')
        except Exception:
            pass

    def forward(self, ids, texts=None):
        if self._adaptive_blend:
            self._adaptive_math_blend()
        x = self.base.embed(ids)
        if texts is not None:
            # Inject math feature per batch
            batch_size = ids.size(0)
            seq_len = ids.size(1)
            math_feats = torch.zeros(batch_size, seq_len, device=x.device)
            for i, t in enumerate(texts):
                parsed = self._parse_math(t)
                if parsed:
                    val, _, _, _ = parsed
                    # Broadcast scalar feature across seq
                    math_feats[i, :] = float(val)
            gate = torch.sigmoid(self.gate)
            # Project scalar to dim via simple scaling
            x = x + gate * math_feats.unsqueeze(-1) * 0.01
        for layer in self.base.layers:
            x = layer(x)
        logits = self.base.head(x)
        # Logit blending for math positions
        if texts is not None:
            logits = self._blend_logits_with_math(logits, texts)
        return logits

    def _blend_logits_with_math(self, logits, texts):
        # For each example, if math present, bias next token toward result token
        batch_size, seq_len, vocab = logits.shape
        device = logits.device
        for i, t in enumerate(texts):
            parsed = self._parse_math(t)
            if not parsed:
                continue
            val, op, _, _ = parsed
            token_id = self._result_token_id(val)
            if token_id is None:
                continue
            # Use per-op blend for finer control
            blend = self._per_op_blend.get(op, self.math_blend)
            bias = torch.zeros(vocab, device=device)
            bias[token_id] = 2.0
            logits[i, -1, :] += blend * bias
        return logits

    def next_token_probs(self, ids, texts=None, temperature=1.0):
        logits = self.forward(ids, texts)
        probs = torch.softmax(logits / temperature, dim=-1)
        return probs
