"""
Migrate existing Third Brain ledger to sharded multi-layer storage
"""
import json
from pathlib import Path

BASE = Path('C:/Users/thoma/.lmstudio/apps/bionic/projects/d49037d8-47f4-5808-9028-c707de117f8f/workspace/TRANSLATION_OS_V1')
LOG_DIR = BASE / 'runtime' / 'logs'
OLD_LEDGER = LOG_DIR / 'third_brain_ledger.jsonl'
SHARD_DIR = LOG_DIR / 'third_brain_ledger_shards'
INDEX_PATH = LOG_DIR / 'third_brain_ledger_index.json'

def migrate():
    if not OLD_LEDGER.exists():
        print('No old ledger found')
        return
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    shards = {}
    count = 0
    with open(OLD_LEDGER, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                e = json.loads(line)
                ts = e.get('ts', 0)
                shard_key = str(int(ts // 3600))
                shards.setdefault(shard_key, []).append(e)
                count += 1
            except Exception:
                pass
    for k, entries in shards.items():
        shard_path = SHARD_DIR / f"ledger_{k}.jsonl"
        with open(shard_path, 'w', encoding='utf-8') as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")
    # Write index
    index = {"shards": sorted(shards.keys())}
    INDEX_PATH.write_text(json.dumps(index))
    print(f'Migrated {count} entries into {len(shards)} shards')

if __name__ == '__main__':
    migrate()
