"""
Central persistence using BitDrop v2
"""
import sys, json
from pathlib import Path
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "infrastructure"))
from bitdrop_v2_integration import compress_data

def save_json(path, obj):
    data = json.dumps(obj).encode("utf-8")
    meta = compress_data(data)
    bd_path = path.with_suffix(path.suffix + ".bd2")
    # Store metadata + original for now
    with open(bd_path, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "data": obj}, f)

def save_line(path, obj):
    data = json.dumps(obj).encode("utf-8")
    meta = compress_data(data)
    bd_path = path.with_suffix(path.suffix + ".bd2")
    with open(bd_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"meta": meta, "data": obj}) + "\n")

def load_json_bitdrop(path):
    bd_path = path.with_suffix(path.suffix + ".bd2")
    if bd_path.exists():
        try:
            # Last written entry
            lines = bd_path.read_text(encoding="utf-8").strip().split("\n")
            if lines and lines[-1]:
                entry = json.loads(lines[-1])
                return entry.get("data")
        except Exception:
            pass
    # Fallback to plain
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except:
            return None
    return None

def load_lines_bitdrop(path):
    bd_path = path.with_suffix(path.suffix + ".bd2")
    if not bd_path.exists():
        return []
    items = []
    try:
        for line in bd_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            items.append(entry.get("data"))
    except Exception:
        return []
    return items
