"""
Hybrid Inference – Weave → Fusion → Fractal → WeaveFormer
"""
import torch
try:
    from .weave_transformer import build_weave_pipeline
    from .symbolic_neural_fusion import SymbolicNeuralFusion
    from .fractal_router import FractalRouter
    from .weaveformer import WeaveFormer
except Exception:
    from weave_transformer import build_weave_pipeline
    from symbolic_neural_fusion import SymbolicNeuralFusion
    from fractal_router import FractalRouter
    from weaveformer import WeaveFormer

class HybridEngine:
    def __init__(self, dim=256, vocab_size=512, persist_path=None):
        # Init KV-Webb runtime for resonance persistence
        kv_webb = None
        try:
            import sys
            from pathlib import Path
            base = Path(__file__).resolve().parent.parent.parent.parent
            sys.path.insert(0, str(base / "infrastructure" / "kv_webb" / "kv_web_runtime"))
            from runtime import KvWebRuntime
            kv_webb = KvWebRuntime()
        except Exception:
            kv_webb = None
        resonance_path = str(base / "runtime" / "logs" / "resonance_memory.pt") if persist_path else None
        self.weave = build_weave_pipeline(dim, kv_webb=kv_webb, persist_path=resonance_path)
        self.fusion = SymbolicNeuralFusion(dim, vocab_size, persist_path)
        self.router = FractalRouter()
        self.weaveformer = WeaveFormer(dim)
        try:
            from .tokenizer import Tokenizer
        except Exception:
            from tokenizer import Tokenizer
        self.tokenizer = Tokenizer()
        import torch.nn as nn
        self.embedding = nn.Embedding(vocab_size, dim)
    
    def inference(self, text):
        # Semantic-as-context: get focused context bundle
        context_text = ""
        try:
            import sys
            from pathlib import Path
            base = Path(__file__).resolve().parent.parent.parent.parent
            sys.path.insert(0, str(base / "infrastructure" / "attention"))
            from context_attention import build_focused_context_bundle
            bundle = build_focused_context_bundle(text, top_k=5)
            candidates = bundle.get('focused_context', [])
            if candidates:
                ctx_parts = [c.get('answer','') for c in candidates if c.get('answer')]
                if ctx_parts:
                    context_text = " | ".join(ctx_parts[:3])
        except Exception:
            context_text = ""
        # Prepend context to query
        enriched_text = f"{context_text} | {text}" if context_text else text
        token_ids = self.tokenizer.encode(enriched_text)
        if not token_ids:
            token_ids = [0]
        ids_tensor = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)
        x = self.embedding(ids_tensor)
        # Update resonance memory and persist to KV-Webb
        _ = self.weave["resonance"](x)
        branch_out = self.weave["branches"](x)
        collapsed = self.weave["collapse"](branch_out)
        fused = self.fusion.fuse(collapsed)
        pts = fused["pts"]
        routed = self.router.route(pts)
        pattern_ids = torch.tensor([p.pattern_id for p in routed], dtype=torch.long).unsqueeze(0)
        logits = self.weaveformer(pattern_ids)
        return {"logits_shape": tuple(logits.shape), "num_patterns": len(routed), "pts_count": len(pts), "context_used": bool(context_text), "context_preview": context_text[:200]}
