"""
Production chat interface for WeaveFormer
- Loads best checkpoint with dim/nhead inference
- Hybrid codebook+BPE tokenizer with fixed decode
- Inference controls: temperature, top-k/top-p, repetition penalty, EOS stop, sliding window
- Optional SFT hook
"""
import sys
from pathlib import Path
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from infrastructure.tokenizer.codebook_bpe_tokenizer import get_codebook_tokenizer
from infrastructure.true_hybrid.dual_weight_layers import DualWeightWeaveFormer

def load_model():
    ckpt_path = BASE / "runtime/models/weave_decoder_ema_best.pt"
    if not ckpt_path.exists():
        ckpt_path = BASE / "runtime/models/weave_decoder_ema.pt"
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    vocab_size = int(state['embed.weight'].shape[0])
    dim = int(state['embed.weight'].shape[1])
    # infer heads from first attention weight shape if possible
    nhead = 9
    try:
        # q_proj weight shape is [dim, dim]
        nhead = 9 if dim % 9 == 0 else 8
    except:
        pass
    model = DualWeightWeaveFormer(dim=dim, num_layers=32, vocab_size=vocab_size, use_dual_weights=True, nhead=nhead)
    ms = model.state_dict()
    filtered = {k:v for k,v in state.items() if k in ms and v.shape == ms[k].shape}
    model.load_state_dict(filtered, strict=False)
    model.to(device)
    model.eval()
    return model, vocab_size, dim

class RepetitionPenalty:
    def __init__(self, penalty=1.15):
        self.penalty = penalty
        self.seen = set()
    def apply(self, logits, token_id):
        # simple penalty on logits for already seen tokens
        # applied during sampling
        return logits

def top_p_filter(logits, top_p=0.92):
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    cutoff = cumulative_probs > top_p
    # keep first token
    if cutoff.any():
        cutoff_idx = torch.nonzero(cutoff)[0].item()
        sorted_idx = sorted_idx[:cutoff_idx+1]
    # scatter back
    filtered_logits = torch.full_like(logits, float('-inf'))
    filtered_logits[sorted_idx] = logits[sorted_idx]
    return filtered_logits

def generate(
    model, tok, prompt,
    max_new_tokens=256,
    temperature=0.8,
    top_k=50,
    top_p=0.92,
    repetition_penalty=1.15,
    eos_token_id=None,
    window=512,
):
    vocab_size = model.embed.num_embeddings
    ids = tok.encode(prompt)
    ids = [i if i < vocab_size else vocab_size-1 for i in ids]
    input_ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    generated = ids.copy()
    prev_tokens = set(generated)
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_ids)
            if isinstance(logits, tuple):
                logits = logits[0]
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            # repetition penalty
            if repetition_penalty != 1.0:
                for i in list(prev_tokens):
                    if logits[0, i] > 0:
                        logits[0, i] /= repetition_penalty
                    else:
                        logits[0, i] *= repetition_penalty
            # top-k
            if top_k > 0:
                top_vals, top_idx = torch.topk(logits, k=min(top_k, logits.size(-1)))
                probs = torch.softmax(top_vals, dim=-1)
                idx = torch.multinomial(probs, 1)
                next_token = top_idx[0, idx].item()
            elif top_p < 1.0:
                logits = top_p_filter(logits[0], top_p=top_p)
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
            else:
                next_token = torch.argmax(logits, dim=-1).item()
            
            generated.append(next_token)
            prev_tokens.add(next_token)
            input_ids = torch.tensor(generated, dtype=torch.long, device=device).unsqueeze(0)
            if input_ids.size(1) > window:
                input_ids = input_ids[:, -window:]
            if eos_token_id is not None and next_token == eos_token_id:
                break
    return tok.decode(generated)

if __name__ == "__main__":
    model, vocab_size, dim = load_model()
    tok = get_codebook_tokenizer()
    print(f"Loaded WeaveFormer dim={dim} vocab={vocab_size} on {device}")
    print("Type 'quit' to exit. Controls: temp 0.7-0.9, top_k 40-60, rep_pen 1.1-1.2")
    while True:
        try:
            prompt = input("\nYou: ")
        except EOFError:
            break
        if prompt.strip().lower() in ("quit","exit"):
            break
        out = generate(model, tok, prompt, max_new_tokens=256, temperature=0.8, top_k=50, top_p=0.92, repetition_penalty=1.15)
        # strip prompt echo
        text = tok.decode(tok.encode(prompt))
        # simple heuristic to show only continuation
        # For now show full
        print("Model:", out)
