"""
Parallel Weave Map with Memory Pool integration.
Links map/ledger/graph/index with weave lattice for parallel orchestrator.
"""
from infrastructure.memory_pool.memory_pool import get_memory_pool
from pathlib import Path
import json, time

BASE = Path(__file__).resolve().parents[2]
RUNTIME_LOGS = BASE / "runtime" / "logs"

class ParallelWeaveMap:
    def __init__(self):
        self.pool = get_memory_pool()
        self.ledger = []
        self.graph = {}
        self.index = {}
    
    def log(self, stage, query, result):
        entry = {
            "ts": time.time(),
            "stage": stage,
            "query": query,
            "response_source": result.get("response_source"),
            "confidence": result.get("confidence",0),
            "response": result.get("response")
        }
        self.ledger.append(entry)
        # Update graph and index
        qkey = str(query)
        self.graph.setdefault(qkey, []).append(entry)
        self.index[qkey] = entry
        # Put into memory pool for parallel consumers
        self.pool.put({"type":"weave_map","stage":stage,"entry":entry})
    
    def get_links(self, query):
        return self.graph.get(str(query), [])
    
    def save(self):
        try:
            log_path = RUNTIME_LOGS / "parallel_weave_map.jsonl"
            with open(log_path, "a", encoding="utf-8") as f:
                for e in self.ledger:
                    f.write(json.dumps(e)+"\n")
        except Exception:
            pass

_weave_map = ParallelWeaveMap()

def get_parallel_weave_map():
    return _weave_map
