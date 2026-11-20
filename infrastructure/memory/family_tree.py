"""
Production-grade Family Tree / Lineage system for AI Memory
Models hierarchical relationships between memories for inheritance and reasoning.
"""
from __future__ import annotations
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
import numpy as np

@dataclass
class FamilyNode:
    id: str
    parent_id: Optional[str] = None
    children: List[str] = field(default_factory=list)
    generation: int = 0
    embedding: Optional[np.ndarray] = None
    metadata: Dict = field(default_factory=dict)

class FamilyTree:
    """
    Hierarchical lineage for memories.
    Supports parent-child links, inheritance of properties, and ancestral queries.
    """
    def __init__(self):
        self.nodes: Dict[str, FamilyNode] = {}
        self.lock = threading.RLock()

    def add_node(self, vid: str, parent_id: Optional[str] = None, embedding: Optional[np.ndarray] = None, metadata: Optional[Dict] = None):
        with self.lock:
            if vid in self.nodes:
                node = self.nodes[vid]
                if embedding is not None:
                    node.embedding = embedding
                if metadata:
                    node.metadata.update(metadata)
                return node
            gen = 0
            if parent_id:
                parent = self.nodes.get(parent_id)
                if parent:
                    gen = parent.generation + 1
                    parent.children.append(vid)
            node = FamilyNode(id=vid, parent_id=parent_id, generation=gen, embedding=embedding, metadata=metadata or {})
            self.nodes[vid] = node
            return node

    def get_ancestors(self, vid: str, max_depth: int = 10) -> List[str]:
        with self.lock:
            ancestors = []
            cur = self.nodes.get(vid)
            depth = 0
            while cur and cur.parent_id and depth < max_depth:
                ancestors.append(cur.parent_id)
                cur = self.nodes.get(cur.parent_id)
                depth += 1
            return ancestors

    def get_descendants(self, vid: str) -> List[str]:
        with self.lock:
            result = []
            stack = [vid]
            while stack:
                cur_id = stack.pop()
                node = self.nodes.get(cur_id)
                if not node:
                    continue
                for child in node.children:
                    result.append(child)
                    stack.append(child)
            return result

    def find_common_ancestor(self, vid1: str, vid2: str) -> Optional[str]:
        with self.lock:
            anc1 = set(self.get_ancestors(vid1, max_depth=20))
            anc1.add(vid1)
            cur = vid2
            while cur:
                if cur in anc1:
                    return cur
                node = self.nodes.get(cur)
                cur = node.parent_id if node else None
            return None

    def link_by_similarity(self, vids: List[str], embeddings: Dict[str, np.ndarray], threshold: float = 0.85) -> int:
        """Automatically link similar memories as parent-child."""
        with self.lock:
            links = 0
            # Sort by generation or creation order heuristic
            for i in range(len(vids)):
                for j in range(i+1, len(vids)):
                    vi = vids[i]
                    vj = vids[j]
                    e1 = embeddings.get(vi)
                    e2 = embeddings.get(vj)
                    if e1 is None or e2 is None:
                        continue
                    sim = float(np.dot(e1, e2) / (np.linalg.norm(e1)*np.linalg.norm(e2)+1e-9))
                    if sim >= threshold:
                        # Link newer as child of older
                        self.add_node(vj, parent_id=vi)
                        links += 1
            return links

    def prune_orphans(self, min_generation: int = 2) -> int:
        with self.lock:
            removed = 0
            for vid, node in list(self.nodes.items()):
                if node.generation >= min_generation and not node.children and node.parent_id is None:
                    del self.nodes[vid]
                    removed += 1
            return removed

    def export(self) -> Dict:
        with self.lock:
            return {
                vid: {
                    "parent_id": n.parent_id,
                    "generation": n.generation,
                    "children": n.children,
                    "metadata": n.metadata,
                }
                for vid, n in self.nodes.items()
            }
