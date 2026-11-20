"""
Experience Epochs - consolidate successful interactions into improved snapshots and master map.
"""
import json, time
from pathlib import Path
from collections import defaultdict

BASE = Path(__file__).resolve().parent.parent.parent.parent
MASTER_MAP_PATH = BASE / "runtime" / "logs" / "master_map.json"
SNAPSHOT_PATH = BASE / "infrastructure" / "snapshot_memory" / "snapshots.json"
EPOCH_LOG = BASE / "runtime" / "logs" / "experience_epochs.json"

def _load_json(p):
    if p.exists():
        try: return json.loads(p.read_text())
        except: return {}
    return {}

def _save_json(p, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2))

class ExperienceEpochs:
    def __init__(self, threshold=50):
        self.threshold = threshold
        self.buffer = []

    def log_success(self, query, answer, source, steps=None):
        self.buffer.append({
            "query": query,
            "answer": answer,
            "source": source,
            "steps": steps or [],
            "ts": time.time()
        })
        if len(self.buffer) >= self.threshold:
            self.run_epoch()

    def run_epoch(self):
        # 1. Collect successful memories
        memories = self.buffer
        # 2. Collect successful reasoning paths
        paths = [m.get("steps",[]) for m in memories]
        # 3. Find repeated patterns
        pattern_counts = defaultdict(int)
        for p in paths:
            key = "|".join(p)
            pattern_counts[key] += 1
        # 4. Generate pattern summaries
        summaries = []
        for pat, cnt in pattern_counts.items():
            summaries.append({"pattern": pat, "count": cnt})
        # 5. Update memory weights & 6. Update resonance scores
        master = _load_json(MASTER_MAP_PATH)
        for m in memories:
            q = m["query"]
            # simple canonical key
            import re
            qcan = re.sub(r'[^\w\s]',' ',q.lower())
            qcan = re.sub(r'\s+',' ',qcan).strip()
            entry = master.get(qcan, {})
            entry["answer"] = m["answer"]
            entry["successes"] = entry.get("successes",0)+1
            entry["last_used"] = time.time()
            entry["sources"] = list(set(entry.get("sources",[])+[m["source"]]))
            master[qcan] = entry
        _save_json(MASTER_MAP_PATH, master)

        # 7. Store improved snapshots
        snaps = _load_json(SNAPSHOT_PATH)
        for m in memories:
            pt = m["query"].split("?")[0][:80]
            key = pt.lower()
            snap = snaps.get(key, {})
            snap["successes"] = snap.get("successes",0)+1
            snap["confidence"] = min(1.0, snap.get("confidence",0.5)+0.05)
            snap["last_used"] = time.time()
            snaps[key] = snap
        _save_json(SNAPSHOT_PATH, snaps)

        # log epoch
        log = _load_json(EPOCH_LOG)
        log.setdefault("epochs", []).append({
            "ts": time.time(),
            "count": len(memories),
            "patterns": summaries
        })
        _save_json(EPOCH_LOG, log)

        self.buffer = []
        return {"processed": len(memories), "patterns_found": len(summaries)}

_exp = ExperienceEpochs()

def log_success(query, answer, source, steps=None):
    _exp.log_success(query, answer, source, steps)

def run_epoch():
    return _exp.run_epoch()
