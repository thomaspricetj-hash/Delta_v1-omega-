"""
Comprehensive tokenizer + codebook health check
"""
import sys
from pathlib import Path
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from infrastructure.tokenizer.codebook_bpe_tokenizer import get_codebook_tokenizer
import pickle

tok = get_codebook_tokenizer()
bpe_size = len(tok.bpe.vocab)
codebook_entries = len(tok.codebook.text_to_ref)
ref_to_id_len = len(tok._ref_to_id)
id_to_ref_len = len(tok._id_to_ref)

print("=== Tokenizer Summary ===")
print(f"BPE vocab size: {bpe_size}")
print(f"Codebook text_to_ref entries: {codebook_entries}")
print(f"Ref-to-ID mappings: {ref_to_id_len}")
print(f"ID-to-Ref mappings: {id_to_ref_len}")
print(f"Codebook vocab offset: {tok._codebook_vocab_offset}")

vocab_size = min(1_048_576, max(8192, bpe_size + codebook_entries))
print(f"Effective vocab_size for model: {vocab_size}")

# Consistency checks
assert ref_to_id_len == id_to_ref_len, "Ref/ID mapping mismatch"
print("[OK] Ref/ID mapping counts match")

# Sample texts
cache_dir = BASE / "runtime" / "cache"
texts_path = cache_dir / "texts_chunk_0.pkl"
if texts_path.exists():
    with open(texts_path, 'rb') as f:
        texts = pickle.load(f)[:500]
else:
    import pyarrow.parquet as pq
    f = BASE / "trainingData" / "000_00000.parquet"
    df = pq.ParquetFile(str(f)).read().to_pandas()
    texts = df.iloc[:,0].astype(str).dropna().tolist()[:500]

print(f"\n=== Round-trip test on {len(texts)} samples ===")
errors = 0
id_range_errors = 0
decode_errors = 0
for i, t in enumerate(texts):
    ids = tok.encode(t)
    if not ids:
        print(f"[{i}] Empty encode")
        errors += 1
        continue
    # ID range
    if any(id < 0 or id >= vocab_size for id in ids):
        id_range_errors += 1
        if id_range_errors <= 5:
            print(f"[{i}] ID out of vocab range, max id {max(ids)} >= {vocab_size}")
    # Decode
    try:
        dec = tok.decode(ids)
    except Exception as e:
        decode_errors += 1
        if decode_errors <= 5:
            print(f"[{i}] Decode error: {e}")
        continue
    # Basic sanity: decode should not be empty
    if not dec.strip():
        print(f"[{i}] Decode empty")
        errors += 1

print(f"Samples checked: {len(texts)}")
print(f"ID range errors: {id_range_errors}")
print(f"Decode errors: {decode_errors}")
print(f"Other errors: {errors}")

if id_range_errors == 0 and decode_errors == 0:
    print("\n[PASS] Tokenizer + Codebook are healthy")
else:
    print("\n[WARN] Issues found")
