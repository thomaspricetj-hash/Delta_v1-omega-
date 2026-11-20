"""
Compress existing shards with BitDrop v2
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path('C:/Users/thoma/.lmstudio/apps/bionic/projects/d49037d8-47f4-5808-9028-c707de117f8f/workspace/TRANSLATION_OS_V1/infrastructure')))
import bitdrop_v2 as bd

BASE = Path('C:/Users/thoma/.lmstudio/apps/bionic/projects/d49037d8-47f4-5808-9028-c707de117f8f/workspace/TRANSLATION_OS_V1')
SHARD_DIR = BASE / 'runtime' / 'embeddings' / 'shards'

for shard_file in sorted(SHARD_DIR.glob('embed_store_shard_*.bd2.bd2')):
    data = shard_file.read_bytes()
    compressed = bd.compress_adaptive_py(data)
    # Write compressed to .bd2 file, keep original as .bak
    bak = shard_file.with_suffix('.bd2.bd2.bak')
    shard_file.rename(bak)
    shard_file.write_bytes(compressed)
    print(f'Compressed {shard_file.name}: {len(data)} -> {len(compressed)} bytes')

print('Done')