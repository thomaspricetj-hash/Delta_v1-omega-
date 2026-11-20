"""
Global default-compressed JSONL logger
All new logs under runtime/logs are written compressed by default.
"""
from pathlib import Path
import json
from .bitdrop_log_writer import get_writer

_writer = get_writer()

def append_jsonl(log_path, obj):
    """Append a JSON object to log_path, compressed via BitDrop."""
    line = json.dumps(obj, ensure_ascii=False)
    _writer.write_line(Path(log_path), line)

def append_lines(log_path, objs):
    lines = [json.dumps(o, ensure_ascii=False) for o in objs]
    _writer.append_lines(Path(log_path), lines)

def iter_jsonl(log_path):
    """Yield Python objects from a compressed JSONL log."""
    for line in _writer.read_lines(Path(log_path)):
        try:
            yield json.loads(line)
        except Exception:
            continue

def exists(log_path):
    return _writer.exists(Path(log_path))
