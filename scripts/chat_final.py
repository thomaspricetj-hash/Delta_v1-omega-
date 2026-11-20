"""
Production chat with robust sampling, temperature annealing, and repetition control.
Loads weave_decoder_main.pt, uses proper softmax + top-p / top-k.
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
    ckpt_path = BASE / "runtime/models/weave_decoder_main.pt"
    if not ckpt_path.exists():
        # fallback
        ckpt_path = BASE / "runtime/models/weave_decoder_best.pt"
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    vocab_size = int(state['embed.weight'].shape[0])
    dim = int(state['embed.weight'].shape[1])
    nhead = 9 if dim % 9 == 0 else 8
    model = DualWeightWeaveFormer(dim=dim, num_layers=32, vocab_size=vocab_size, use_dual_weights=True, nhead=nhead)
    ms = model.state_dict()
    filtered = {k:v for k,v in state.items() if k in ms and v.shape == ms[k].shape}
    model.load_state_dict(filtered, strict=False)
    model.to(device)
    model.eval()
    return model, vocab_size, dim

def top_p_logits(logits, top_p=0.92):
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    probs = torch.softmax(sorted_logits, dim=-1)
    cum_probs = torch.cumsum(probs, dim=-1)
    cutoff = cum_probs > top_p
    if cutoff.any():
        cutoff_idx = torch.nonzero(cutoff)[0].item()
        sorted_idx = sorted_idx[:cutoff_idx+1]
    # scatter back
    filtered = torch.full_like(logits, float('-inf'))
    filtered[sorted_idx] = logits[sorted_idx]
    return filtered

def generate(model, tok, prompt, max_new_tokens=256, temperature=0.75, top_p=0.92, top_k=0, rep_penalty=1.1, window=512):
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
            logits = logits[0, -1, :] / max(temperature, 1e-8)
            # repetition penalty
            if rep_penalty != 1.0:
                for i in list(prev_tokens):
                    if logits[i] > 0:
                        logits[i] /= rep_penalty
                    else:
                        logits[i] *= rep_penalty
            # top-p / top-k
            if top_p < 1.0:
                logits = top_p_logits(logits, top_p=top_p)
            elif top_k > 0:
                # top-k sampling
                top_vals, top_idx = torch.topk(logits, k=min(top_k, logits.size(-1)))
                probs = torch.softmax(top_vals, dim=-1)
                idx = torch.multinomial(probs, 1)
                next_token = top_idx[idx].item()
            else:
                next_token = torch.argmax(logits).item()
            if top_p < 1.0 or top_k == 0:
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
            generated.append(next_token)
            prev_tokens.add(next_token)
            input_ids = torch.tensor(generated[-window:], dtype=torch.long, device=device).unsqueeze(0)
            # stop on EOS if present
            if next_token == 0:
                break
    return tok.decode(generated)

if __name__ == "__main__":
    model, vocab_size, dim = load_model()
    tok = get_codebook_tokenizer()
    print(f"Loaded WeaveFormer dim={dim} vocab={vocab_size} on {device}")
    print("Controls: temp 0.6-0.9, top_p 0.9-0.95, rep_penalty 1.05-1.2")
    while True:
        try:
            prompt = input("\nYou: ")
        except EOFError:
            break
        if prompt.strip().lower() in ("quit","exit"):
            break
        out = generate(model, tok, prompt, max_new_tokens=256, temperature=0.75, top_p=0.92, top_k=0, rep_penalty=1.1)
        print("Model:", out)
