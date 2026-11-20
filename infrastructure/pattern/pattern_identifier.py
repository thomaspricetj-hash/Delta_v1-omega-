"""
Production-grade Pattern Identifier for TRANSLATION_OS_V1
Detects recurring token motifs, tracks frequency/recency, and provides sampling weights.
Thread-safe, SIMD-friendly, integrates with MBFN/HSCL and curriculum.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import threading
import hashlib
from collections import defaultdict, deque
import numpy as np
from pathlib import Path
from infrastructure.true_hybrid.tree_store import TreeStore

@dataclass
class PatternStat:
    id: str
    count: int = 0
    last_seen: int = 0
    first_seen: int = 0
    examples: List[str] = field(default_factory=list)

class PatternIdentifier:
    """
    Online pattern discovery for token windows.
    Uses rolling hash for n-gram motifs and LSH-style bucketing.
    """
    def __init__(self, ngram_sizes: Tuple[int, ...] = (3,4,5,8), decay: float = 0.999, tree_dir: str = "runtime/memory/pattern_tree"):
        self.ngram_sizes = ngram_sizes
        self.decay = decay
        self.lock = threading.RLock()
        self.patterns: Dict[str, PatternStat] = {}
        self.step = 0
        # Recent pattern buffer for fast recall
        self.recent = deque(maxlen=8192)
        # TreeStore for unbounded pattern growth
        self._tree = TreeStore(capacity_per_node=10000, dir_path=tree_dir, name="patterns")
        self._load_patterns_from_tree()

    def _load_patterns_from_tree(self):
        # Load persisted patterns from tree nodes for incremental resume
        try:
            for node in self._tree._walk():
                for entry in node.entries:
                    # entries stored as JSON strings
                    import json
                    try:
                        data = json.loads(entry)
                        pid = data.get('id')
                        if pid:
                            self.patterns[pid] = PatternStat(
                                id=pid,
                                count=int(data.get('count',0)),
                                last_seen=int(data.get('last_seen',0)),
                                first_seen=int(data.get('first_seen',0)),
                                examples=data.get('examples',[])
                            )
                    except Exception:
                        continue
        except Exception:
            pass

    def _persist_pattern(self, pid: str):
        # Persist single pattern to tree for incremental save
        stat = self.patterns.get(pid)
        if not stat:
            return
        import json
        payload = json.dumps({
            'id': stat.id,
            'count': stat.count,
            'last_seen': stat.last_seen,
            'first_seen': stat.first_seen,
            'examples': stat.examples
        })
        node = self._tree._find_leaf()
        if len(node.entries) >= self._tree.capacity:
            self._tree._split_node(node)
            node = self._tree._find_leaf()
        # Replace if exists
        for i,e in enumerate(node.entries):
            try:
                if json.loads(e).get('id') == pid:
                    node.entries[i] = payload
                    node.save()
                    return
            except Exception:
                pass
        node.entries.append(payload)
        node.save()

    def _hash_window(self, window: List[int], n: int) -> str:
        if len(window) < n:
            return ""
        # Fast rolling hash via tuple + sha256 for stability
        # Use first n-gram only for speed; can be extended to all n-grams
        key = tuple(window[:n])
        h = hashlib.blake2b(str(key).encode(), digest_size=16).hexdigest()
        return f"n{n}_{h}"

    def identify(self, window: List[int]) -> List[str]:
        ids = []
        for n in self.ngram_sizes:
            pid = self._hash_window(window, n)
            if pid:
                ids.append(pid)
        return ids

    def update(self, window: List[int], step: int | None = None):
        if step is not None:
            self.step = step
        ids = self.identify(window)
        with self.lock:
            for pid in ids:
                stat = self.patterns.get(pid)
                if stat is None:
                    stat = PatternStat(id=pid, first_seen=self.step)
                    self.patterns[pid] = stat
                stat.count += 1
                stat.last_seen = self.step
                if len(stat.examples) < 3:
                    stat.examples.append(str(window[:min(16, len(window))]))
                self.recent.append(pid)
                # Persist incrementally
                self._persist_pattern(pid)
            # Decay old counts
            if self.step % 1000 == 0:
                for s in self.patterns.values():
                    s.count = int(s.count * self.decay) 

    def get_weight(self, window: List[int]) -> float:
        """Return sampling weight based on pattern rarity.
        Rare patterns get higher weight to improve coverage.
        """
        ids = self.identify(window)
        if not ids:
            return 1.0
        with self.lock:
            scores = []
            for pid in ids:
                stat = self.patterns.get(pid)
                if stat is None:
                    scores.append(2.0)  # unseen = high weight
                else:
                    # Inverse frequency with recency boost
                    freq = max(1, stat.count)
                    recency = max(0, self.step - stat.last_seen)
                    score = 1.0 / np.sqrt(freq) * (1.0 + 0.01 * recency)
                    scores.append(score)
            return float(np.mean(scores))

    def top_patterns(self, k: int = 10) -> List[Tuple[str, int]]:
        with self.lock:
            items = sorted(self.patterns.items(), key=lambda x: x[1].count, reverse=True)
            return [(pid, stat.count) for pid, stat in items[:k]]

    def stats(self) -> Dict[str, int]:
        with self.lock:
            return {
                "patterns": len(self.patterns),
                "recent_buffer": len(self.recent),
                "step": self.step,
            }
