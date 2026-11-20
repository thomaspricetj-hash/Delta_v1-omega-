"""
Migrate all orchestrator ledgers to sharded multi-layer storage
"""
import json
from pathlib import Path

BASE = Path('C:/Users/thoma/.lmstudio/apps/bionic/projects/d49037d8-47f4-5808-9028-c707de117f8f/workspace/TRANSLATION_OS_V1')
LOG_DIR = BASE / 'runtime' / 'logs'

LEDGERS = [
    'predictor_convergence.jsonl',
    'orchestrator_diagnostics.jsonl',
    'third_brain_ledger.jsonl',
    'day_dream_queue.jsonl',
    'foldable_answer_ledger.jsonl',
    'evidence_ledger.jsonl',
    'fast_path_stats.jsonl',
]

def shard_key(ts):
    return str(int(ts // 3600))

def migrate_one(name):
    path = LOG_DIR / name
    if not path.exists():
        print(f'Skip {name} - not found')
        return
    shard_dir = LOG_DIR / f'{name}_shards'
    shard_dir.mkdir(parents=True, exist_ok=True)
    shards = {}
    count = 0
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                e = json.loads(line)
                ts = e.get('ts', 0)
                key = shard_key(ts)
                shards.setdefault(key, []).append(e)
                count += 1
            except Exception:
                pass
    for k, entries in shards.items():
        shard_path = shard_dir / f'{k}.jsonl'
        with open(shard_path, 'w', encoding='utf-8') as f:
            for e in entries:
                f.write(json.dumps(e) + '\n')
    # Write index
    index_path = LOG_DIR / f'{name}_index.json'
    index_path.write_text(json.dumps({"shards": sorted(shards.keys())}))
    print(f'Migrated {name}: {count} entries into {len(shards)} shards')

if __name__ == '__main__':
    for name in LEDGERS:
        migrate_one(name)
