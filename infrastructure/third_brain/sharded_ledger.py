"""
Multi-layered sharded ledger for Third Brain
"""
import json
from pathlib import Path
import hashlib
import time

BASE = Path(__file__).resolve().parents[2].parent.parent
LOG_DIR = BASE / "runtime" / "logs"
LEDGER_DIR = LOG_DIR / "third_brain_ledger_shards"
INDEX_PATH = LOG_DIR / "third_brain_ledger_index.json"

def _ensure_dir():
    LEDGER_DIR.mkdir(parents=True, exist_ok=True)

def _shard_key(ts):
    # Simple time-based sharding
    return str(int(ts // 3600))  # per hour

def append(entry):
    _ensure_dir()
    ts = entry.get('ts', 0)
    shard = _shard_key(ts)
    shard_path = LEDGER_DIR / f"ledger_{shard}.jsonl"
    with open(shard_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(entry) + "\n")
    # Update weave-linked index
    idx = {}
    if INDEX_PATH.exists():
        try:
            idx = json.loads(INDEX_PATH.read_text(encoding='utf-8'))
        except Exception:
            pass
    shards = idx.get('shards', [])
    if shard not in shards:
        # Build weave links
        weave_node = {
            'shard': shard,
            'ts': ts,
            'prev': shards[-1] if shards else None,
            'next': None,
            'weave_fingerprint': hashlib.sha256((shard+str(ts)).encode()).hexdigest()[:16]
        }
        # Update previous shard's next pointer
        if shards:
            prev_key = shards[-1]
            # Load existing weave links
            weave_links = idx.get('weave_links', {})
            if prev_key in weave_links:
                weave_links[prev_key]['next'] = shard
            else:
                weave_links[prev_key] = {'next': shard}
        # Store new node
        weave_links = idx.get('weave_links', {})
        weave_links[shard] = weave_node
        idx['shards'] = shards + [shard]
        idx['weave_links'] = weave_links
    INDEX_PATH.write_text(json.dumps(idx))
