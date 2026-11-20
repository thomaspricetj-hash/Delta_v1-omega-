"""
Ingest latest-all.json.bz2 into trainingData as parquet shards, streaming.
Usage: python scripts/ingest_wiki_to_parquet.py
"""
import bz2
import json
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
TRAIN_DATA_DIR = BASE / "trainingData"
TRAIN_DATA_DIR.mkdir(parents=True, exist_ok=True)

SRC = Path(r"C:\Users\thoma\Downloads\latest-all.json.bz2")
OUT_PREFIX = "wiki"
CHUNK_SIZE = 50000

def iter_json_records(bz2_path):
    with bz2.open(bz2_path, 'rt', encoding='utf-8', errors='ignore') as f:
        # Detect JSONL vs single JSON
        first = None
        for line in f:
            line = line.strip()
            if line:
                first = line
                break
        if not first:
            return
        f.seek(0)
        if first.startswith('{'):
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception:
                    continue
        else:
            # Try load whole file as JSON array
            try:
                data = json.load(f)
                if isinstance(data, list):
                    for rec in data:
                        yield rec
                else:
                    yield data
            except Exception:
                f.seek(0)
                for line in f:
                    try:
                        yield json.loads(line)
                    except Exception:
                        continue

def extract_text(rec):
    if isinstance(rec, dict):
        parts = []
        for k in ['text','content','article','body','page_content','title','summary','text_content']:
            v = rec.get(k)
            if isinstance(v, str) and v.strip():
                parts.append(v.strip())
        if not parts:
            # fallback to all string values
            for v in rec.values():
                if isinstance(v, str) and len(v) > 20:
                    parts.append(v.strip())
        return ' '.join(parts)
    if isinstance(rec, str):
        return rec
    return ''

def main():
    if not SRC.exists():
        raise FileNotFoundError(f"Source not found: {SRC}")
    print(f"Reading {SRC}")
    chunk = []
    shard_idx = 0
    total = 0
    max_records = 1000000  # set to None for full ingest
    for i, rec in enumerate(iter_json_records(SRC)):
        if max_records and i >= max_records:
            break
        txt = extract_text(rec)
        if txt:
            chunk.append({'text': txt})
        if len(chunk) >= CHUNK_SIZE:
            out_path = TRAIN_DATA_DIR / f"{OUT_PREFIX}-{shard_idx:04d}.parquet"
            pd.DataFrame(chunk).to_parquet(out_path, index=False)
            print(f"Wrote {out_path} with {len(chunk)} rows")
            total += len(chunk)
            chunk = []
            shard_idx += 1
        if i % 100000 == 0 and i>0:
            print(f"Processed {i} records, total written {total}")
    if chunk:
        out_path = TRAIN_DATA_DIR / f"{OUT_PREFIX}-{shard_idx:04d}.parquet"
        pd.DataFrame(chunk).to_parquet(out_path, index=False)
        print(f"Wrote {out_path} with {len(chunk)} rows")
        total += len(chunk)
    print(f"Done. Total rows written: {total}")

if __name__ == "__main__":
    main()
