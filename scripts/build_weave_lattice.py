"""
Build multilayered weave lattice linking shards via weave index layers
"""
import json
from pathlib import Path

BASE = Path('C:/Users/thoma/.lmstudio/apps/bionic/projects/d49037d8-47f4-5808-9028-c707de117f8f/workspace/TRANSLATION_OS_V1')
STORE_DIR = BASE / 'runtime' / 'embeddings'
INDEX_PATH = STORE_DIR / 'embed_store_index.json'
MAP_PATH = STORE_DIR / 'embed_store_map.json'
WEAVE_INDEX_PATH = STORE_DIR / 'weave_index.json'
LATTICE_PATH = STORE_DIR / 'embed_store_weave_lattice.json'

index = {}
if INDEX_PATH.exists():
    index = json.loads(INDEX_PATH.read_text(encoding='utf-8'))

map_data = {}
if MAP_PATH.exists():
    map_data = json.loads(MAP_PATH.read_text(encoding='utf-8'))

weave = {}
if WEAVE_INDEX_PATH.exists():
    weave = json.loads(WEAVE_INDEX_PATH.read_text(encoding='utf-8')).get('index', {})

# Build shard -> layers mapping
shard_layers = {}
shard_keys = {}
for key, shard_id in index.items():
    shard_id = str(shard_id)
    shard_keys.setdefault(shard_id, set()).add(key)
    meta = weave.get(key, {})
    layer = meta.get('layer', 0)
    shard_layers.setdefault(shard_id, {}).setdefault(layer, set()).add(key)

# Build multilayered lattice
layers = {}
for shard_id, layer_map in shard_layers.items():
    for layer, keys in layer_map.items():
        layers.setdefault(layer, {}).setdefault(shard_id, set()).update(keys)

# Build edges: shards connected if they share layer keys
shard_ids = list(shard_keys.keys())
edges = []
for i in range(len(shard_ids)):
    for j in range(i+1, len(shard_ids)):
        s1 = shard_ids[i]
        s2 = shard_ids[j]
        common = shard_keys[s1].intersection(shard_keys[s2])
        if common:
            edges.append({'from': s1, 'to': s2, 'shared_keys': len(common)})

lattice = {
    'shards': {sid: {'entries': len(keys)} for sid, keys in shard_keys.items()},
    'layers': {str(l): {sid: list(set(keys)) for sid, keys in layer_map.items()} for l, layer_map in layers.items()},
    'edges': edges,
    'total_shards': len(shard_keys)
}

with open(LATTICE_PATH, 'w', encoding='utf-8') as f:
    json.dump(lattice, f, indent=2)

print(f'Weave lattice built with {len(shard_keys)} shards, {len(layers)} layers, {len(edges)} edges')
print(f'Saved to {LATTICE_PATH}')