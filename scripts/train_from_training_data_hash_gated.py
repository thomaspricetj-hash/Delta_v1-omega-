"""
Hash-gated training data ingest to avoid re-processing identical chunks
"""
import hashlib
import json
from pathlib import Path

BASE = Path('C:/Users/thoma/.lmstudio/apps/bionic/projects/d49037d8-47f4-5808-9028-c707de117f8f/workspace/TRANSLATION_OS_V1')
TRAIN_DIR = BASE.parent / 'training data'
INDEX_PATH = BASE / 'runtime' / 'training_data_index.json'

def load_index():
    if INDEX_PATH.exists():
        return json.loads(INDEX_PATH.read_text())
    return {}

def save_index(idx):
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    INDEX_PATH.write_text(json.dumps(idx))

def hash_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()

def main():
    idx = load_index()
    for p in TRAIN_DIR.rglob('*.*'):
        if not p.is_file():
            continue
        h = hash_file(p)
        if h in idx:
            # Skip, just bump replay count
            idx[h]['replay_count'] += 1
            idx[h]['last_seen'] = __import__('time').time()
            continue
        # New file: process
        # Placeholder: call teach_full via orchestrator
        # For demo, just record
        idx[h] = {
            'path': str(p),
            'hash': h,
            'first_seen': __import__('time').time(),
            'replay_count': 1
        }
    save_index(idx)
    print(f'Processed training data, unique files: {len(idx)}')

if __name__ == '__main__':
    main()
