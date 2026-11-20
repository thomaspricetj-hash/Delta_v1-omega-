"""
WeaveFormer autoregressive decoder on top of embeddings.
Uses existing WeaveFormer + PredictiveLinear with matrix-multiply predictor offload.
Simple greedy / temperature sampling.
"""
import torch
import torch.nn.functional as F
from pathlib import Path
import sys
import hashlib
from infrastructure.predictors.predictor_base import PredictorCache, log_predictor_event, log_reward

BASE = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(BASE))
from infrastructure.true_hybrid.weaveformer import WeaveFormer

sys.path.insert(0, str(BASE))
from infrastructure.tokenizer.bpe_tokenizer import get_bpe_tokenizer

class WeaveDecoder:
    def __init__(self, dim=256, num_layers=2, vocab_size=8192, device='cpu'):
        self.device = device
        self.tokenizer = get_bpe_tokenizer()
        # Build vocab mapping from tokenizer
        vocab = getattr(self.tokenizer, 'vocab', {})
        # If vocab is dict token->id, invert
        if isinstance(vocab, dict):
            self.id2tok = {v:k for k,v in vocab.items()}
            self.vocab_size = len(vocab)
        else:
            self.id2tok = {}
            self.vocab_size = vocab_size
        # Clamp vocab size to reasonable limit
        self.vocab_size = min(self.vocab_size, vocab_size)
        self.model = WeaveFormer(dim=dim, num_layers=num_layers, vocab_size=self.vocab_size).to(self.device)
        self.model.eval()
        # Enable predictor mode
        self.model.enable_predictor_mode()

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=64, temperature=0.8, top_k=40):
        # Encode prompt
        try:
            ids = self.tokenizer.encode(prompt)
        except Exception:
            ids = []
        if not ids:
            ids = [0]
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        generated = ids
        for _ in range(max_new_tokens):
            out = self.model(input_ids)
            logits = out[0] if isinstance(out, tuple) else out
            # Take last token logits
            next_logits = logits[0, -1, :]
            if temperature > 0:
                next_logits = next_logits / temperature
            # Top-k filtering
            if top_k > 0:
                top_k_vals, top_k_idx = torch.topk(next_logits, min(top_k, self.vocab_size))
                probs = F.softmax(top_k_vals, dim=-1)
                idx = torch.multinomial(probs, 1).item()
                next_token = top_k_idx[idx].item()
            else:
                next_token = torch.argmax(next_logits, dim=-1).item()
            generated.append(next_token)
            # Stop on eos if defined
            # Append to input
            input_ids = torch.tensor([generated], dtype=torch.long, device=self.device)
            # Simple stop condition for demo
            if next_token == 0:
                break
        # Decode
        try:
            text = self.tokenizer.decode(generated)
        except Exception:
            text = prompt
        return text

# Convenience function with predictor-style cache
_decoder_instance = None
_decoder_cache = PredictorCache("weave_decoder")

def _make_cache_key(prompt, max_new_tokens, temperature, top_k):
    key_str = f"{prompt}|{max_new_tokens}|{temperature}|{top_k}"
    return hashlib.sha256(key_str.encode()).hexdigest()

def generate_with_weave(prompt, max_new_tokens=64, temperature=0.8, top_k=40):
    global _decoder_instance
    if _decoder_instance is None:
        _decoder_instance = WeaveDecoder()
    cache_key = _make_cache_key(prompt, max_new_tokens, temperature, top_k)
    cached_val, hit = _decoder_cache.get(cache_key)
    if hit:
        log_predictor_event("weave_decoder", "cache_hit", {"prompt_len": len(prompt)})
        return cached_val
    # Generate
    out = _decoder_instance.generate(prompt, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
    _decoder_cache.set(cache_key, out)
    # Log reward based on length / confidence proxy
    confidence = min(0.99, 0.5 + 0.5 * min(1.0, len(out)/200))
    reward = 0.5 + 0.5 * confidence
    log_reward("weave_decoder", reward, confidence, query=prompt[:200])
    log_predictor_event("weave_decoder", "predict", {"prompt_len": len(prompt), "out_len": len(out), "confidence": confidence})
    return out
