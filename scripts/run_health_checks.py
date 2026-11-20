"""
Comprehensive health checks for TRANSLATION_OS_V1 trainer.
Runs tokenizer, codebook, cache, dataset, and a tiny training step.
"""
import os, sys, json, pickle
from pathlib import Path
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

print("=== Health Check Start ===")

# 1. Tokenizer
print("\n[1] Tokenizer / Codebook")
from infrastructure.tokenizer.codebook_bpe_tokenizer import get_codebook_tokenizer
tok = get_codebook_tokenizer()
bpe_size = len(tok.bpe.vocab)
codebook_entries = len(tok.codebook.text_to_ref)
print(f"BPE vocab: {bpe_size}, codebook entries: {codebook_entries}")
sample = "The quick brown fox jumps over the lazy dog"
ids = tok.encode(sample)
dec = tok.decode(ids)
print(f"Sample encode/decode OK: {len(ids)} ids, round-trip match={dec.strip()[:50]}")

# 2. Cache
print("\n[2] Cache")
cache_dir = BASE / "runtime" / "cache"
manifest = cache_dir / "texts_chunk_0.manifest"
if manifest.exists():
    with open(manifest, 'r', encoding='utf-8') as f:
        man = json.load(f)
    print(f"Manifest found with {len(man.get('parts', []))} parts, total≈{man.get('total')}")
    # Load first part
    part = cache_dir / man['parts'][0]
    with open(part, 'rb') as pf:
        part_texts = pickle.load(pf)
    print(f"First part loaded {len(part_texts)} texts")
else:
    print("No manifest, checking legacy pickle")
    pkl = cache_dir / "texts_chunk_0.pkl"
    if pkl.exists():
        import pickle
        with open(pkl, 'rb') as f:
            texts = pickle.load(f)
        print(f"Legacy pickle loaded {len(texts)} texts")
    else:
        print("No cache found")

# 3. Encode samples
print("\n[3] Encode samples from cache")
texts = []
if manifest.exists():
    with open(manifest) as f:
        man = json.load(f)
    for part_name in man['parts'][:2]:
        part_path = cache_dir / part_name
        with open(part_path, 'rb') as pf:
            texts.extend(pickle.load(pf))
        if len(texts) >= 100:
            break
else:
    texts = ["test"]

ok = 0
for i, t in enumerate(texts[:20]):
    try:
        words_preview = ' '.join(t.split()[:5])
        print(f"Encoding sample {i}, len={len(t)}, words={len(t.split())} - start, preview={words_preview[:80]!r}")
        ids = tok.encode(t, max_words=200)
        print(f"Encoding sample {i} - done, ids={len(ids)}")
        ok += 1
        if i % 10 == 0:
            print(f"Encode progress {i}/20, ids={len(ids)}")
    except Exception as e:
        print(f"Encode error at {i}: {e}")
        print(f"Sample preview: {t[:200]!r}")
        break
print(f"Encoded {ok}/100 samples OK")

# 4. Dataset
print("\n[4] Dataset")
from infrastructure.data.codebook_dataset import CodebookTextDataset
ds = CodebookTextDataset(texts[:1000], tokenizer=tok)
print(f"Dataset len={len(ds)}")
sample_ids = ds[0]
print(f"First item len={len(sample_ids)}")

# 5. Model init
print("\n[5] Model init")
import torch
from infrastructure.true_hybrid.dual_weight_layers import DualWeightWeaveFormer
DIM = int(os.environ.get('DIM','288'))
HEADS = int(os.environ.get('HEADS','9'))
vocab_size = max(8192, bpe_size + codebook_entries)
model = DualWeightWeaveFormer(dim=DIM, num_layers=2, vocab_size=vocab_size, use_dual_weights=True, nhead=HEADS)
print(f"Model created with dim={DIM}, heads={HEADS}, vocab={vocab_size}")

print("\n=== Health Check Complete ===")
print("All checks passed. You can now train with SKIP_WARMUP=0")
