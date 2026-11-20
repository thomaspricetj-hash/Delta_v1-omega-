import os, sys, json, pickle
from pathlib import Path
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from infrastructure.tokenizer.codebook_bpe_tokenizer import get_codebook_tokenizer

tok = get_codebook_tokenizer()
cache_dir = BASE / "runtime" / "cache"
manifest = cache_dir / "texts_chunk_0.manifest"
if not manifest.exists():
    print("No manifest found")
    sys.exit(0)

with open(manifest, 'r', encoding='utf-8') as f:
    man = json.load(f)

parts = man.get('parts', [])
texts = []
for part_file in parts[:2]:
    part_path = cache_dir / part_file
    with open(part_path, 'rb') as pf:
        texts.extend(pickle.load(pf))
    if len(texts) >= 100:
        break

print(f"Testing encode on {len(texts)} samples")
for i, t in enumerate(texts):
    try:
        ids = tok.encode(t)
        if i % 10 == 0:
            print(f"[{i}] ok len={len(ids)}")
    except Exception as e:
        print(f"[{i}] ERROR {e}")
        print(f"Sample: {t[:200]!r}")
        break
print("Done")
