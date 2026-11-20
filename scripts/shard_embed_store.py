"""
Shard embed_store.bd2.bd2 into ~100MB BitDrop shards with index/map
"""
import json
from pathlib import Path

BASE = Path('C:/Users/thoma/.lmstudio/apps/bionic/projects/d49037d8-47f4-5808-9028-c707de117f8f/workspace/TRANSLATION_OS_V1')
STORE_DIR = BASE / 'runtime' / 'embeddings'
SOURCE = STORE_DIR / 'embed_store.bd2.bd2'
SHARD_DIR = STORE_DIR / 'shards'
SHARD_DIR.mkdir(exist_ok=True)

INDEX_PATH = STORE_DIR / 'embed_store_index.json'
MAP_PATH = STORE_DIR / 'embed_store_map.json'

shards = {}
index = {}
shard_id = 0
buf = []
buf_size = 0
MAX_SHARD_BYTES = 100 * 1024 * 1024  # 100MB

def flush_shard():
    global shard_id, buf, buf_size
    if not buf:
        return
    shard_file = SHARD_DIR / f'embed_store_shard_{shard_id}.bd2.bd2'
    with open(shard_file, 'w', encoding='utf-8') as f:
        for obj in buf:
            f.write(json.dumps(obj)+'\n')
    shards[shard_id] = str(shard_file)
    buf = []
    buf_size = 0
    shard_id += 1

print('Reading source...')
with open(SOURCE, 'r', encoding='utf-8') as f:
    for line in f:
        if not line.strip():
            continue
        obj = json.loads(line)
        data = obj.get('data', {})
        key = data.get('key')
        if key:
            index[key] = shard_id
        buf.append(obj)
        buf_size += len(line)
        if buf_size >= MAX_SHARD_BYTES:
            flush_shard()

flush_shard()
# Save index and map
with open(INDEX_PATH, 'w', encoding='utf-8') as f:
    json.dump(index, f)
with open(MAP_PATH, 'w', encoding='utf-8') as f:
    json.dump({'shards': shards, 'count': len(index)}, f)

print(f'Created {shard_id} shards')
print(f'Index entries: {len(index)}')
print('Done')