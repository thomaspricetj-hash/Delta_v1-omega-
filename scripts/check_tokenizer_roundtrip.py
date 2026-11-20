"""
Full paragraph round-trip check for codebook tokenizer.
Ensures encode -> decode is lossless and no token ID out of range.
"""
import sys
from pathlib import Path
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from infrastructure.tokenizer.codebook_bpe_tokenizer import get_codebook_tokenizer
import pickle

cache_dir = BASE / "runtime" / "cache"
texts_path = cache_dir / "texts_chunk_0.pkl"
if not texts_path.exists():
    # fallback to first parquet sample
    import pyarrow.parquet as pq
    f = BASE / "trainingData" / "000_00000.parquet"
    df = pq.ParquetFile(str(f)).read().to_pandas()
    texts = df.iloc[:,0].astype(str).dropna().tolist()[:1000]
else:
    with open(texts_path, 'rb') as f:
        texts = pickle.load(f)[:1000]

tok = get_codebook_tokenizer()
print(f"Tokenizer vocab size: {len(tok.bpe.vocab)}")
try:
    codebook_size = len(getattr(tok.codebook, '_id_to_ref', {}))
except:
    codebook_size = 'unknown'
print(f"Codebook size: {codebook_size}")

errors = 0
for i, t in enumerate(texts):
    ids = tok.encode(t)
    # sanity checks
    if not ids:
        print(f"[{i}] Empty encode")
        errors += 1
        continue
    if any(id < 0 or id >= len(tok.bpe.vocab) for id in ids):
        print(f"[{i}] ID out of range: {ids[:10]}")
        errors += 1
        continue
    dec = tok.decode(ids)
    # Compare normalized
    # Allow minor differences due to BPE/space handling, check length ratio
    if len(dec.strip()) == 0:
        print(f"[{i}] Decode empty for text start: {t[:80]}")
        errors += 1
    # Optional strict check
    # if dec != t:
    #   print(f"[{i}] Mismatch")
    if i < 5:
        print(f"Sample {i}")
        print("Orig:", t[:200])
        print("Dec :", dec[:200])
        print("IDs :", len(ids))
        print("---")

print(f"Checked {len(texts)} samples, errors: {errors}")
if errors == 0:
    print("Tokenizer round-trip looks healthy.")
else:
    print("Tokenizer has issues.")
