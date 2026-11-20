"""
Build a top-N word dictionary for CodebookBPETokenizer
"""
import sys
from pathlib import Path
import json
import pickle
from collections import Counter

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

N = int(sys.argv[1]) if len(sys.argv) > 1 else 50000

cache_dir = BASE / "runtime" / "cache"
texts_path = cache_dir / "texts_chunk_0.pkl"
texts = []
if texts_path.exists():
    with open(texts_path, 'rb') as f:
        texts = pickle.load(f)
else:
    # Fallback: sample from first parquet
    import pyarrow.parquet as pq
    f = BASE / "trainingData" / "000_00000.parquet"
    df = pq.ParquetFile(str(f)).read().to_pandas()
    texts = df.iloc[:,0].astype(str).dropna().tolist()

print(f"Loaded {len(texts)} texts for dictionary build")

counter = Counter()
for t in texts:
    # Simple word split
    words = t.split()
    counter.update(words)

most_common = counter.most_common(N)
word_to_id = {}
bpe_vocab_size = 8192  # will be offset by tokenizer
for i, (word, _) in enumerate(most_common):
    word_to_id[word] = bpe_vocab_size + i

out_path = BASE / "runtime" / "token_dictionary.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump({"word_to_id": word_to_id}, f)

print(f"Saved dictionary with {len(word_to_id)} words to {out_path}")
