import os
from pathlib import Path
import pickle
import json
import torch
import sys
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
from infrastructure.tokenizer.codebook_bpe_tokenizer import get_codebook_tokenizer

cache_dir = BASE / 'runtime' / 'cache'
chunk_idx = int(os.environ.get('CHUNK_INDEX','0'))
manifest = cache_dir / f'texts_chunk_{chunk_idx}.manifest'
if not manifest.exists():
    print(f'Manifest not found: {manifest}')
    sys.exit(1)
with open(manifest, 'r', encoding='utf-8') as f:
    man = json.load(f)
parts = man.get('parts', [])
texts = []
for p in parts:
    with open(cache_dir / p, 'rb') as pf:
        texts.extend(pickle.load(pf))
print(f'Loaded {len(texts)} texts from manifest')
tok = get_codebook_tokenizer()
max_texts = int(os.environ.get('MAX_TEXTS_FOR_PRETRAIN','0')) or None
if max_texts:
    texts = texts[:max_texts]
print(f'Pre-tokenizing {len(texts)} texts...')
tokenized = []
for i, t in enumerate(texts):
    if i % 1000 == 0:
        print(f'  {i}/{len(texts)}')
    ids = tok.encode(t)
    tokenized.append(ids)
tokens_path = cache_dir / f'tokens_chunk_{chunk_idx}.pt'
torch.save(tokenized, tokens_path)
meta = {'num_texts': len(texts)}
with open(cache_dir / f'tokens_chunk_{chunk_idx}_meta.json', 'w') as f:
    json.dump(meta, f)
print(f'Saved tokens to {tokens_path}')
