"""
Pattern Store & Transformation Reuse Engine
Store patterns once, store transformations, reuse transformations.
Reduces recomputation by recognizing repeated weight/pattern blocks.
"""
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any
import json
from pathlib import Path

@dataclass
class PatternEntry:
    pattern_id: str
    data: List[float]
    usage_count: int = 0
    transformations: Dict[str, 'TransformationEntry'] = field(default_factory=dict)

@dataclass
class TransformationEntry:
    transform_id: str
    operation: str
    params: Dict[str, Any]
    correction: List[float]

class PatternStore:
    def __init__(self, persist_path: Path = None):
        self.patterns: Dict[str, PatternEntry] = {}
        self.transform_cache: Dict[Tuple[str,str], str] = {}
        self.persist_path = persist_path
        self._load()

    def _hash(self, data):
        h = hashlib.sha256()
        h.update(json.dumps(data, sort_keys=True).encode())
        return h.hexdigest()[:16]

    def store_pattern(self, data: List[float]) -> str:
        pid = self._hash(data)
        if pid not in self.patterns:
            self.patterns[pid] = PatternEntry(pattern_id=pid, data=data)
        self.patterns[pid].usage_count += 1
        try:
            from common_index.common_index import index_pattern
            index_pattern(pid)
        except Exception:
            pass
        return pid

    def get_pattern(self, pid: str) -> PatternEntry:
        return self.patterns.get(pid)

    def store_transformation(self, pattern_id: str, operation: str, params: Dict, correction: List[float]) -> str:
        tid = self._hash((operation, params, correction))
        if pattern_id not in self.patterns:
            raise ValueError("Pattern not stored")
        entry = self.patterns[pattern_id]
        if tid not in entry.transformations:
            entry.transformations[tid] = TransformationEntry(
                transform_id=tid,
                operation=operation,
                params=params,
                correction=correction
            )
        key = (pattern_id, tid)
        self.transform_cache[key] = tid
        try:
            from common_index.common_index import index_transformation
            index_transformation(tid)
        except Exception:
            pass
        return tid

    def reuse_transformation(self, pattern_id: str, operation: str, params: Dict) -> Tuple[bool, List[float]]:
        # Find existing transformation with same operation/params
        if pattern_id not in self.patterns:
            return False, []
        entry = self.patterns[pattern_id]
        for t in entry.transformations.values():
            if t.operation == operation and t.params == params:
                return True, t.correction
        return False, []

    def apply_pattern(self, pattern_id: str, transform_id: str = None):
        pat = self.patterns.get(pattern_id)
        if not pat:
            return None
        if not transform_id:
            return pat.data
        trans = pat.transformations.get(transform_id)
        if not trans:
            return pat.data
        # Pattern + small correction
        result = [p + c for p,c in zip(pat.data, trans.correction)]
        return result

    def save(self):
        if not self.persist_path:
            return
        data = {
            "patterns": {
                pid: {
                    "data": p.data,
                    "usage_count": p.usage_count,
                    "transformations": {
                        tid: {
                            "operation": t.operation,
                            "params": t.params,
                            "correction": t.correction
                        } for tid,t in p.transformations.items()
                    }
                } for pid,p in self.patterns.items()
            }
        }
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.persist_path, 'w') as f:
            json.dump(data, f)

    def _load(self):
        if self.persist_path and self.persist_path.exists():
            with open(self.persist_path) as f:
                data = json.load(f)
            for pid, v in data.get("patterns", {}).items():
                entry = PatternEntry(pattern_id=pid, data=v["data"], usage_count=v["usage_count"])
                for tid, tv in v["transformations"].items():
                    entry.transformations[tid] = TransformationEntry(
                        transform_id=tid,
                        operation=tv["operation"],
                        params=tv["params"],
                        correction=tv["correction"]
                    )
                self.patterns[pid] = entry

# Singleton
_store = PatternStore(Path("TRANSLATION_OS_V1/infrastructure/pattern_store/pattern_store.json"))

def store_pattern(data):
    return _store.store_pattern(data)

def store_transformation(pid, operation, params, correction):
    return _store.store_transformation(pid, operation, params, correction)

def reuse_transformation(pid, operation, params):
    return _store.reuse_transformation(pid, operation, params)

def apply_pattern(pid, tid=None):
    return _store.apply_pattern(pid, tid)

def save_store():
    _store.save()
