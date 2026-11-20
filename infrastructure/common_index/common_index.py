"""
Common Index
Index of:
- Common patterns
- Common matrix blocks
- Common transformations
- Common outputs
Enables fast lookup instead of recomputation.
"""
import json
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict

class CommonIndex:
    def __init__(self, persist_path: Path = None):
        self.patterns = defaultdict(int)
        self.matrix_blocks = defaultdict(int)
        self.transformations = defaultdict(int)
        self.outputs = defaultdict(int)
        self.reasoning_paths = defaultdict(int)
        self.persist_path = persist_path
        self._load()

    def index_pattern(self, pid: str):
        self.patterns[pid] += 1

    def index_matrix_block(self, bid: str):
        self.matrix_blocks[bid] += 1

    def index_transformation(self, tid: str):
        self.transformations[tid] += 1

    def index_output(self, oid: str):
        self.outputs[oid] += 1

    def index_reasoning_path(self, rid: str):
        self.reasoning_paths[rid] += 1

    def get_common(self, n=10):
        return {
            "patterns": sorted(self.patterns.items(), key=lambda x: x[1], reverse=True)[:n],
            "matrix_blocks": sorted(self.matrix_blocks.items(), key=lambda x: x[1], reverse=True)[:n],
            "transformations": sorted(self.transformations.items(), key=lambda x: x[1], reverse=True)[:n],
            "outputs": sorted(self.outputs.items(), key=lambda x: x[1], reverse=True)[:n],
            "reasoning_paths": sorted(self.reasoning_paths.items(), key=lambda x: x[1], reverse=True)[:n],
        }

    def save(self):
        if not self.persist_path:
            return
        data = {
            "patterns": dict(self.patterns),
            "matrix_blocks": dict(self.matrix_blocks),
            "transformations": dict(self.transformations),
            "outputs": dict(self.outputs),
            "reasoning_paths": dict(self.reasoning_paths),
        }
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.persist_path, 'w') as f:
            json.dump(data, f)

    def _load(self):
        if self.persist_path and self.persist_path.exists():
            with open(self.persist_path) as f:
                data = json.load(f)
            self.patterns = defaultdict(int, data.get("patterns", {}))
            self.matrix_blocks = defaultdict(int, data.get("matrix_blocks", {}))
            self.transformations = defaultdict(int, data.get("transformations", {}))
            self.outputs = defaultdict(int, data.get("outputs", {}))
            self.reasoning_paths = defaultdict(int, data.get("reasoning_paths", {}))

# Singleton
_index = CommonIndex(Path("TRANSLATION_OS_V1/infrastructure/common_index/common_index.json"))

def get_index():
    return _index

def index_pattern(pid):
    _index.index_pattern(pid)

def index_matrix_block(bid):
    _index.index_matrix_block(bid)

def index_transformation(tid):
    _index.index_transformation(tid)

def index_output(oid):
    _index.index_output(oid)

def index_reasoning_path(rid):
    _index.index_reasoning_path(rid)

def save_index():
    _index.save()
