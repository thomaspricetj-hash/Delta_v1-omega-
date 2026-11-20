"""
Multilayered Foldable Map with Index/Ledger for answer overlays
Right/wrong answers can be overlayed to find convergence
"""
import json
from pathlib import Path
import time

BASE = Path(__file__).resolve().parents[2].parent.parent
LOG_DIR = BASE / "runtime" / "logs"
MAP_PATH = LOG_DIR / "foldable_answer_map.json"
INDEX_PATH = LOG_DIR / "foldable_answer_index.json"
LEDGER_PATH = LOG_DIR / "foldable_answer_ledger.jsonl"

def _load_map():
    try:
        if MAP_PATH.exists():
            return json.loads(MAP_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {"layers": {"correct": {}, "wrong": {}, "overlay": {}}}

def _save_map(m):
    try:
        MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
        MAP_PATH.write_text(json.dumps(m, ensure_ascii=False))
    except Exception:
        pass

def log_answer(query, answer, is_correct, source=None):
    qk = query.lower().strip()
    m = _load_map()
    layers = m.setdefault("layers", {})
    correct = layers.setdefault("correct", {})
    wrong = layers.setdefault("wrong", {})
    overlay = layers.setdefault("overlay", {})

    entry = {
        "answer": answer,
        "source": source,
        "ts": time.time()
    }
    if is_correct:
        correct[qk] = entry
    else:
        wrong[qk] = entry

    # Update overlay convergence
    if qk in correct and qk in wrong:
        overlay[qk] = {
            "correct": correct[qk]["answer"],
            "wrong": wrong[qk]["answer"],
            "convergence": "diverge",
            "ts": time.time()
        }
    elif qk in correct:
        overlay[qk] = {"correct": correct[qk]["answer"], "convergence": "known"}

    _save_map(m)

    # Ledger - BitDrop compressed
    try:
        from infrastructure.logging.bitdrop_log_writer import get_writer
        writer = get_writer()
        LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps({
            "ts": time.time(),
            "query": qk,
            "answer": answer,
            "is_correct": is_correct,
            "source": source
        })
        writer.write_line(LEDGER_PATH, line)
    except Exception:
        # Fallback to plain
        try:
            LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(LEDGER_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "ts": time.time(),
                    "query": qk,
                    "answer": answer,
                    "is_correct": is_correct,
                    "source": source
                }) + "\n")
        except Exception:
            pass

    # Index update
    try:
        idx = {}
        if INDEX_PATH.exists():
            idx = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
        idx.setdefault(qk, {"correct": False, "wrong": False})
        idx[qk]["correct"] = qk in correct
        idx[qk]["wrong"] = qk in wrong
        idx[qk]["last_ts"] = time.time()
        INDEX_PATH.write_text(json.dumps(idx, ensure_ascii=False))
    except Exception:
        pass

def get_overlay(query):
    qk = query.lower().strip()
    m = _load_map()
    return m.get("layers", {}).get("overlay", {}).get(qk)
