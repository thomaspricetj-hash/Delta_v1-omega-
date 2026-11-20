"""
Quick talk test for WeaveFormer
Loads latest checkpoint and generates a short continuation.
"""
import sys
from pathlib import Path
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

from infrastructure.tokenizer.codebook_bpe_tokenizer import get_codebook_tokenizer
from infrastructure.true_hybrid.dual_weight_layers import DualWeightWeaveFormer

tok = get_codebook_tokenizer()

ckpt_path = BASE / "runtime" / "models" / "weave_decoder_ema_best.pt"
if not ckpt_path.exists():
    ckpt_path = BASE / "runtime" / "models" / "weave_decoder_ema.pt"
if not ckpt_path.exists():
    ckpt_path = BASE / "runtime" / "models" / "weave_decoder_full.pt"
print(f"Loading checkpoint {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location='cpu')
state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
# Infer vocab size and dim from checkpoint embed weight
if 'embed.weight' in state:
    vocab_size = state['embed.weight'].shape[0]
    dim = state['embed.weight'].shape[1]
else:
    dim = 256
    # fallback
    # fallback
    vocab = getattr(tok,'bpe').vocab
    bpe_size = len(vocab)
    codebook_entries = len(getattr(tok.codebook,'text_to_ref',{}))
    vocab_size = min(1_048_576, max(8192, bpe_size + codebook_entries))
print(f"Using vocab_size from checkpoint: {vocab_size}, dim={dim}")

model = DualWeightWeaveFormer(dim=dim, num_layers=32, vocab_size=vocab_size, use_dual_weights=True)
# Filter by shape
model_state = model.state_dict()
filtered = {}
for k,v in state.items():
    if k in model_state and v.shape == model_state[k].shape:
        filtered[k]=v
    else:
        print(f"Skip {k} shape mismatch {v.shape} vs {model_state[k].shape}")
model.load_state_dict(filtered, strict=False)
model.to(device)
model.eval()

def generate(prompt, max_new_tokens=64, temperature=1.0, top_k=40):
    ids = tok.encode(prompt)
    ids = [i if i < vocab_size else vocab_size-1 for i in ids]
    input_ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    generated = ids.copy()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_ids)
            if isinstance(logits, tuple):
                logits = logits[0]
            logits = logits[:, -1, :]
            if temperature != 1.0:
                logits = logits / temperature
            # top-k
            if top_k > 0:
                top_vals, top_idx = torch.topk(logits, k=min(top_k, logits.size(-1)))
                probs = torch.softmax(top_vals, dim=-1)
                idx = torch.multinomial(probs, 1)
                next_token = top_idx[0, idx].item()
            else:
                next_token = torch.argmax(logits, dim=-1).item()
            generated.append(next_token)
            input_ids = torch.tensor(generated, dtype=torch.long, device=device).unsqueeze(0)
            # keep window
            if input_ids.size(1) > 512:
                input_ids = input_ids[:, -512:]
    text = tok.decode(generated)
    return text

prompts = [
    "The quick brown fox",
    "In the future,",
    "Once upon a time",
]
for p in prompts:
    out = generate(p, max_new_tokens=48, temperature=0.9, top_k=50)
    print("---")
    print("Prompt:", p)
    # safe print for Windows console
    print("Output:", repr(out[:200]))
