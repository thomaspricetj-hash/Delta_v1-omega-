"""
Migrate 33GB embed_store.jsonl.bak to BitDrop v2 sharded store without re-training
"""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path('C:/Users/thoma/.lmstudio/apps/bionic/projects/d49037d8-47f4-5808-9028-c707de117f8f/workspace/TRANSLATION_OS_V1/infrastructure')))
import bitdrop_v2 as bd

BASE = Path('C:/Users/thoma/.lmstudio/apps/bionic/projects/d49037d8-47f4-5808-9028-c707de117f8f/workspace/TRANSLATION_OS_V1')
STORE_DIR = BASE / 'runtime' / 'embeddings'
JSONL_PATH = STORE_DIR / 'embed_store.jsonl.bak'
SHARD_DIR = STORE_DIR / 'shards'
SHARD_DIR.mkdir(exist_ok=True)

INDEX_PATH = STORE_DIR / 'embed_store_index.json'
LEDGER_PATH = STORE_DIR / 'embed_store_ledger.jsonl'
MAP_PATH = STORE_DIR / 'embed_store_map.json'

ENTRIES_PER_SHARD = 10000

index = {}
ledger_entries = []
shards_meta = {}

shard_id = 0
buf = []

def flush_shard(sid, buffer):
    global shard_id
    # Build JSON lines with meta/data wrapper
    lines = []
    for obj in buffer:
        # obj is raw entry from jsonl
        wrapped = {"meta": {}, "data": obj}
        lines.append(json.dumps(wrapped))
    raw_bytes = '\n'.join(lines).encode('utf-8')
    compressed = bd.compress_adaptive_py(raw_bytes)
    shard_file = SHARD_DIR / f'embed_store_shard_{sid}.bd2.bd2'
    shard_file.write_bytes(compressed)
    # Update index/ledger
    meta = {'file': str(shard_file), 'entries': len(buffer), 'keys': []}
    for obj in buffer:
        key = obj.get('key')
        if key:
            index[key] = sid
            ledger_entries.append({'key': key, 'shard_id': sid, 'line_num': 0, 'access_count': 0})
            meta['keys'].append(key)
    shards_meta[sid] = meta
    print(f'Shard {sid} written: {len(buffer)} entries, {len(compressed)} bytes compressed')

print('Streaming JSONL...')
with open(JSONL_PATH, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        buf.append(obj)
        if len(buf) >= ENTRIES_PER_SHARD:
            flush_shard(shard_id, buf)
            shard_id += 1
            buf = []
        if i % 100000 == 0:
            print(f'Processed {i} lines...')

if buf:
    flush_shard(shard_id, buf)

# Write index
with open(INDEX_PATH, 'w', encoding='utf-8') as f:
    json.dump(index, f)

with open(LEDGER_PATH, 'w', encoding='utf-8') as f:
    for e in ledger_entries:
        f.write(json.dumps(e)+'\n')

map_data = {
    'shards': shards_meta,
    'total_shards': len(shards_meta),
    'total_keys': len(index)
}
with open(MAP_PATH, 'w', encoding='utf-8') as f:
    json.dump(map_data, f)

print(f'Done. Created {len(shards_meta)} shards, {len(index)} keys indexed')
