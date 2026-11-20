"""
Shard embed_store into precise manageable shards with index/ledger/graph
"""
import json
from pathlib import Path

BASE = Path('C:/Users/thoma/.lmstudio/apps/bionic/projects/d49037d8-47f4-5808-9028-c707de117f8f/workspace/TRANSLATION_OS_V1')
STORE_DIR = BASE / 'runtime' / 'embeddings'
SOURCE = STORE_DIR / 'embed_store.bd2.bd2'
SHARD_DIR = STORE_DIR / 'shards'
SHARD_DIR.mkdir(exist_ok=True)

INDEX_PATH = STORE_DIR / 'embed_store_index.json'
LEDGER_PATH = STORE_DIR / 'embed_store_ledger.jsonl'
MAP_PATH = STORE_DIR / 'embed_store_map.json'
GRAPH_PATH = STORE_DIR / 'embed_store_graph.json'

ENTRIES_PER_SHARD = 5000

index = {}
ledger_entries = []
shards_meta = {}

shard_id = 0
buf = []

def flush_shard(sid, buffer):
    shard_file = SHARD_DIR / f'embed_store_shard_{sid}.bd2.bd2'
    with open(shard_file, 'w', encoding='utf-8') as f:
        for obj in buffer:
            f.write(json.dumps(obj)+'\n')
    meta = {
        'file': str(shard_file),
        'entries': len(buffer),
        'keys': []
    }
    for i, obj in enumerate(buffer):
        data = obj.get('data', {})
        key = data.get('key')
        if key:
            index[key] = sid
            ledger_entries.append({
                'key': key,
                'shard_id': sid,
                'line_num': i,
                'access_count': 0
            })
            meta['keys'].append(key)
    shards_meta[sid] = meta

print('Sharding embed_store...')
count = 0
with open(SOURCE, 'r', encoding='utf-8') as f:
    for line in f:
        if not line.strip():
            continue
        obj = json.loads(line)
        buf.append(obj)
        count += 1
        if count >= ENTRIES_PER_SHARD:
            flush_shard(shard_id, buf)
            shard_id += 1
            buf = []
            count = 0

if buf:
    flush_shard(shard_id, buf)

# Write index
with open(INDEX_PATH, 'w', encoding='utf-8') as f:
    json.dump(index, f)

# Write ledger
with open(LEDGER_PATH, 'w', encoding='utf-8') as f:
    for entry in ledger_entries:
        f.write(json.dumps(entry)+'\n')

# Write map
map_data = {
    'shards': shards_meta,
    'total_shards': len(shards_meta),
    'total_keys': len(index)
}
with open(MAP_PATH, 'w', encoding='utf-8') as f:
    json.dump(map_data, f)

# Build simple graph
graph = {
    'nodes': [{'id': sid, 'entries': meta['entries']} for sid, meta in shards_meta.items()],
    'edges': []
}
with open(GRAPH_PATH, 'w', encoding='utf-8') as f:
    json.dump(graph, f)

print(f'Created {len(shards_meta)} shards')
print(f'Total keys indexed: {len(index)}')
print('Done')