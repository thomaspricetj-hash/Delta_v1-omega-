"""
Tree-backed DeltaVectorStore with fixed-size nodes and auto-split.
Replaces unbounded JSON save with incremental TreeStore persistence.
"""
from pathlib import Path
import numpy as np
from infrastructure.memory.vector_store import DeltaVector, DeltaVectorStore
from infrastructure.true_hybrid.tree_store import TreeStore, TreeNode

class TreeDeltaVectorStore:
    def __init__(self, dim, capacity_per_node=5000, dir_path="runtime/memory/vecmem_tree"):
        self.dim = dim
        self.dir = Path(dir_path)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.store = TreeStore(capacity_per_node, str(self.dir), "vectors")
        self.in_memory = {}
        # Load existing nodes into memory for fast access
        self._load_all()

    @property
    def vectors(self):
        # Compatibility shim for code expecting .vectors attribute
        return self.in_memory

    def _load_all(self):
        for node in self.store._walk():
            # Node entries are stored as pickled dicts
            for entry in node.entries:
                # entry format: (vid, base_bytes, deltas_bytes, version)
                # Simplified: store as dict
                pass

    def add(self, vid, vector):
        v = np.asarray(vector, dtype=np.float32)
        if v.shape[0] != self.dim:
            raise ValueError("dim mismatch")
        # Store in memory
        self.in_memory[vid] = DeltaVector(id=vid, base=v.copy())
        # Persist to tree
        node = self.store._find_leaf()
        payload = {'vid':vid,'base':v.tolist(),'deltas':[],'version':0}
        if len(node.entries) >= self.store.capacity:
            self.store._split_node(node)
            node = self.store._find_leaf()
        node.entries.append(payload)
        node.save()

    def update(self, vid, delta):
        if vid not in self.in_memory:
            raise KeyError(vid)
        self.in_memory[vid].apply_delta(np.asarray(delta,dtype=np.float32))
        # Update tree node
        for node in self.store._walk():
            for i,e in enumerate(node.entries):
                if e.get('vid')==vid:
                    e['deltas'].append(delta.tolist())
                    e['version']=self.in_memory[vid].version
                    node.save()
                    return
        # not found, append
        node = self.store._find_leaf()
        node.entries.append({'vid':vid,'base':[],'deltas':[delta.tolist()],'version':self.in_memory[vid].version})
        node.save()

    def get(self, vid):
        if vid in self.in_memory:
            return self.in_memory[vid].materialize()
        # fallback to tree
        for node in self.store._walk():
            for e in node.entries:
                if e.get('vid')==vid:
                    base = np.array(e['base'],dtype=np.float32)
                    dv = DeltaVector(id=vid,base=base)
                    for d in e.get('deltas',[]):
                        dv.apply_delta(np.array(d,dtype=np.float32))
                    self.in_memory[vid]=dv
                    return dv.materialize()
        raise KeyError(vid)

    def save(self):
        # TreeStore nodes already saved incrementally
        pass

    def load(self, path):
        # TreeStore loads on init
        pass
