from pathlib import Path
import json

BASE = Path(__file__).resolve().parent.parent.parent
STORE = BASE / 'runtime' / 'logs' / 'cross_reference.jsonl'

def cross_reference(parent_q, child_q, child_a):
    try:
        entry = {
            'parent': parent_q,
            'child': child_q,
            'answer': child_a
        }
        with open(STORE, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry) + '\n')
    except Exception:
        pass
