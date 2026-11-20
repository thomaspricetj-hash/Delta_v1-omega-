"""
Weighted Snapshot Memory for Reasoning Paths
Stores successful problem -> reasoning path snapshots with weights
"""
import time, json
from pathlib import Path
from collections import defaultdict

STORE_PATH = Path(__file__).parent / "snapshots.json"

def _load():
    if STORE_PATH.exists():
        try:
            return json.loads(STORE_PATH.read_text())
        except:
            return {}
    return {}

def _save(data):
    STORE_PATH.write_text(json.dumps(data, indent=2))

class SnapshotMemory:
    def __init__(self):
        self.data = _load()
    
    def store(self, problem_type, steps, confidence=0.9, success=True, memory_paths=None, reasoning_pattern=None, attention_focus=None):
        key = problem_type.lower()
        entry = {
            "problem_type": problem_type,
            "steps": steps,
            "memory_paths": memory_paths or [],
            "reasoning_pattern": reasoning_pattern or [],
            "attention_focus": attention_focus or [],
            "confidence": confidence,
            "success": success,
            "successes": 1,
            "accuracy": 1.0 if success else 0.0,
            "last_used": time.time(),
            "success_score": confidence if success else 0.0
        }
        if key in self.data:
            e = self.data[key]
            e["successes"] += 1
            e["accuracy"] = (e["accuracy"]*(e["successes"]-1) + (1.0 if success else 0.0))/e["successes"]
            e["confidence"] = max(e["confidence"], confidence)
            e["last_used"] = time.time()
            # merge cause info
            if memory_paths:
                e["memory_paths"] = list(set(e.get("memory_paths",[]) + memory_paths))
            if reasoning_pattern:
                e["reasoning_pattern"] = list(set(e.get("reasoning_pattern",[]) + reasoning_pattern))
            if attention_focus:
                e["attention_focus"] = list(set(e.get("attention_focus",[]) + attention_focus))
            e["success_score"] = e.get("success_score",0.0) * 0.9 + (confidence if success else 0.0) * 0.1
            # update steps if new
            if steps:
                e["steps"] = steps
        else:
            self.data[key] = entry
        _save(self.data)

    def retrieve(self, problem_type):
        key = problem_type.lower()
        entry = self.data.get(key)
        if not entry:
            return None
        # Update recency
        entry["last_used"] = time.time()
        _save(self.data)
        return entry

    def weight(self, problem_type):
        entry = self.data.get(problem_type.lower())
        if not entry:
            return 0.0
        successes = entry["successes"]
        accuracy = entry["accuracy"]
        # recency decay
        age = time.time() - entry["last_used"]
        recency = 0.5 ** (age / 86400)  # half-life 1 day
        w = successes * accuracy * recency
        return w

    def best_for(self, problem_type):
        # Simple pattern match: exact or substring
        best = None
        best_w = -1
        for k, v in self.data.items():
            if problem_type.lower() in k or k in problem_type.lower():
                w = self.weight(k)
                if w > best_w:
                    best_w = w
                    best = v
        return best

_snapshot_memory = SnapshotMemory()

def get_snapshot_memory():
    return _snapshot_memory
