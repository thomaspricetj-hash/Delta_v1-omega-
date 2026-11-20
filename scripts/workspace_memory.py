"""
Simple Workspace Memory for persistent cross-turn state
"""
import json
from pathlib import Path
BASE = Path(__file__).resolve().parent.parent
MEM_PATH = BASE / 'runtime' / 'logs' / 'workspace_memory.json'

def load():
    if MEM_PATH.exists():
        try:
            return json.loads(MEM_PATH.read_text())
        except:
            pass
    return {"diary": [], "journal": [], "preferences": {}}

def save(mem):
    MEM_PATH.write_text(json.dumps(mem))

def update_preference(key, value):
    mem = load()
    mem.setdefault("preferences", {})[key] = value
    save(mem)

def get_preference(key):
    mem = load()
    return mem.get("preferences", {}).get(key)

def log_journal(entry):
    mem = load()
    mem.setdefault("journal", []).append({"entry": entry})
    save(mem)
