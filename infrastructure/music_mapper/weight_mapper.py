"""
Weight Mapper – sonify transformer weight updates
Maps gradient / weight delta statistics to musical notes for observability.
Cheap to run: only computes norms and logs a single note per layer per step.
"""
import json
from pathlib import Path
import hashlib
import numpy as np

BASE = Path(__file__).resolve().parents[3]
WEIGHT_LOG = BASE / "runtime" / "logs" / "weight_notes.jsonl"
NOTE_BASE = 48  # G2
NOTE_RANGE = 36 # up to ~G5

def weight_to_note(layer_name, delta_norm, weight_norm):
    # Deterministic base from layer name
    h = int(hashlib.sha256(layer_name.encode()).hexdigest(), 16)
    base_note = NOTE_BASE + (h % NOTE_RANGE)
    # Delta norm shifts pitch: larger update = higher pitch
    # Map delta_norm to -4..+4 semitones
    # Normalize delta_norm heuristically to 0-1 via tanh
    pitch_offset = int(np.tanh(delta_norm * 2.0) * 4)
    note = base_note + pitch_offset
    # Duration from weight magnitude
    duration = 0.2 + 0.6 * min(1.0, weight_norm / 10.0)
    return note, duration

def log_weight_update(layer_name, delta_norm, weight_norm, step):
    note, duration = weight_to_note(layer_name, delta_norm, weight_norm)
    entry = {
        "ts": __import__('time').time(),
        "step": step,
        "layer": layer_name,
        "delta_norm": float(delta_norm),
        "weight_norm": float(weight_norm),
        "note": int(note),
        "duration": float(duration)
    }
    WEIGHT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(WEIGHT_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return entry

def get_weight_harmony():
    if not WEIGHT_LOG.exists():
        return {}
    notes = []
    with open(WEIGHT_LOG, "r", encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            notes.append(e["note"])
    if not notes:
        return {}
    from collections import Counter
    cnt = Counter(notes)
    return {
        "total_updates": len(notes),
        "unique_notes": len(cnt),
        "most_common": cnt.most_common(5)
    }
