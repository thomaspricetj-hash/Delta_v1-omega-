"""
Weave-lattice linked sharded store for Third Brain
Shards are braided via multilayer weave lattice with index/ledger/map
"""
import json
import time
from pathlib import Path
from collections import defaultdict

BASE = Path('C:/Users/thoma/.lmstudio/apps/bionic/projects/d49037d8-47f4-5808-9028-c707de117f8f/workspace/TRANSLATION_OS_V1')
STORE_DIR = BASE / 'runtime' / 'third_brain' / 'shards'
INDEX_PATH = BASE / 'runtime' / 'third_brain' / 'weave_index.json'
LEDGER_PATH = BASE / 'runtime' / 'third_brain' / 'shard_ledger.jsonl'

STORE_DIR.mkdir(parents=True, exist_ok=True)

SHARD_SIZE = 10000

class ShardedWeaveStore:
    def __init__(self):
        self.index = {}
        if INDEX_PATH.exists():
            try:
                self.index = json.loads(INDEX_PATH.read_text())
            except Exception:
                self.index = {}
        self.current_shard = self.index.get('current_shard', 0)
        self.current_count = self.index.get('current_count', 0)

    def _shard_path(self, shard_id):
        return STORE_DIR / f'shard_{shard_id:04d}.jsonl.b2'

    def _save_index(self):
        INDEX_PATH.write_text(json.dumps(self.index))

    def append(self, entry):
        # Roll shard if needed
        if self.current_count >= SHARD_SIZE:
            self.current_shard += 1
            self.current_count = 0
            self.index['current_shard'] = self.current_shard
            self.index['current_count'] = self.current_count
            self._save_index()
        shard_path = self._shard_path(self.current_shard)
        # Append with BitDrop v2 placeholder
        with open(shard_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry)+'\n')
        self.current_count += 1
        self.index['current_count'] = self.current_count
        # Update weave lattice links
        key = entry.get('key')
        if key:
            if 'weave' not in self.index:
                self.index['weave'] = {}
            weave = self.index['weave']
            if key not in weave:
                weave[key] = {'shards': [], 'layers': []}
            weave[key]['shards'].append(self.current_shard)
            # Braid layers: simple layer = shard_id % 8
            layer = self.current_shard % 8
            if layer not in weave[key]['layers']:
                weave[key]['layers'].append(layer)
        # Ledger entry
        with open(LEDGER_PATH, 'a', encoding='utf-8') as lf:
            lf.write(json.dumps({'ts': time.time(), 'shard': self.current_shard, 'key': entry.get('key')} )+'\n')
        self._save_index()

    def load_recent(self, key=None, limit=1000):
        # Load from weave index
        results = []
        if not self.index.get('weave'):
            return results
        if key:
            shards = self.index['weave'].get(key, {}).get('shards', [])
        else:
            # all shards
            shards = list(set([s for v in self.index.get('weave',{}).values() for s in v.get('shards',[]) ]))
        for sid in shards[-5:]:  # last 5 shards
            path = self._shard_path(sid)
            if not path.exists():
                continue
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        if key is None or obj.get('key') == key:
                            results.append(obj)
                            if len(results) >= limit:
                                return results
                    except Exception:
                        pass
        return results

def get_store():
    return ShardedWeaveStore()
