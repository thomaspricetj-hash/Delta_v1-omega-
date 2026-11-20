"""
Evidence Map
Tracks support/oppose evidence for queries and predictions
"""
import json
from pathlib import Path
import time

BASE = Path(__file__).resolve().parents[2].parent.parent
LOG_DIR = BASE / "runtime" / "logs"
EVIDENCE_MAP_PATH = LOG_DIR / "evidence_map.json"
EVIDENCE_LEDGER_PATH = LOG_DIR / "evidence_ledger.jsonl"

def _load_map():
    try:
        if EVIDENCE_MAP_PATH.exists():
            return json.loads(EVIDENCE_MAP_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {"queries": {}}

def _save_map(m):
    try:
        EVIDENCE_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
        EVIDENCE_MAP_PATH.write_text(json.dumps(m, ensure_ascii=False))
    except Exception:
        pass

def log_evidence(query, response, source, confidence, support=1, oppose=0, metadata=None, log_prob=None):
    qk = query.lower().strip()
    m = _load_map()
    qentry = m.setdefault("queries", {}).setdefault(qk, {"evidence": [], "support_sum": 0, "oppose_sum": 0})
    ev = {
        "ts": time.time(),
        "response": response,
        "source": source,
        "confidence": confidence,
        "log_prob": log_prob,
        "support": support,
        "oppose": oppose,
        "metadata": metadata or {}
    }
    qentry["evidence"].append(ev)
    qentry["support_sum"] += support
    qentry["oppose_sum"] += oppose
    # Keep last 200
    qentry["evidence"] = qentry["evidence"][-200:]
    _save_map(m)
    # Ledger
    try:
        EVIDENCE_LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(EVIDENCE_LEDGER_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": time.time(), "query": qk, "response": response, "source": source, "confidence": confidence, "log_prob": log_prob, "support": support, "oppose": oppose}) + "\n")
    except Exception:
        pass

def get_evidence_score(query):
    qk = query.lower().strip()
    m = _load_map()
    qentry = m.get("queries", {}).get(qk)
    if not qentry:
        return None
    sup = qentry.get("support_sum", 0)
    opp = qentry.get("oppose_sum", 0)
    total = sup + opp + 1e-9
    return {"support": sup, "oppose": opp, "score": sup/total}
