"""
Pre-tokenize training texts to runtime/cache/tokens.pt
Run once to eliminate repeated parquet reads + tokenization during training.
"""
import os
from pathlib import Path
import pickle
import torch
from concurrent.futures import ThreadPoolExecutor

BASE = Path(__file__).resolve().parent.parent
sys_path_inserted = False
import sys
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from infrastructure.tokenizer.codebook_bpe_tokenizer import get_codebook_tokenizer

cache_dir = BASE / "runtime" / "cache"
cache_dir.mkdir(parents=True, exist_ok=True)
texts_path = cache_dir / "texts.pkl"
tokens_path = cache_dir / "tokens.pt"
meta_path = cache_dir / "tokens_meta.json"

if not texts_path.exists():
    print("Texts cache not found. Run trainer once with USE_TEXT_CACHE=1 to build texts.pkl")
    sys.exit(0)

with open(texts_path, 'rb') as f:
    texts = pickle.load(f)

print(f"Loaded {len(texts)} texts")

tok = get_codebook_tokenizer()

def encode_one(t):
    try:
        return tok.encode(t)
    except Exception:
        return []

max_workers = int(os.environ.get('PRETOKEN_WORKERS','24'))
print(f"Pre-tokenizing with {max_workers} workers...")
with ThreadPoolExecutor(max_workers=max_workers) as ex:
    tokenized = list(ex.map(encode_one, texts))

# Save as list of lists
import json
meta = {"num_texts": len(texts), "vocab_size": len(tok)}
with open(meta_path, 'w') as f:
    json.dump(meta, f)

torch.save(tokenized, tokens_path)
print(f"Saved tokenized corpus to {tokens_path}")
print("Now set USE_TOKEN_CACHE=1 in trainer to use pre-tokenized data.")
