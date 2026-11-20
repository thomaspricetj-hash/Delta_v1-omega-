"""
Recursive reasoning context window
"""
import json
from pathlib import Path
BASE = Path(__file__).resolve().parent.parent
CTX_PATH = BASE / 'runtime' / 'logs' / 'recursive_context.json'

def load():
    if CTX_PATH.exists():
        return json.loads(CTX_PATH.read_text())
    return {}

def save(d):
    CTX_PATH.write_text(json.dumps(d))

def update(topic, depth):
    d = load()
    d[topic] = depth
    save(d)

def get_depth(topic):
    d = load()
    return d.get(topic, 0)
