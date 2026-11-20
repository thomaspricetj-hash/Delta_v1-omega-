"""
SyntheticMind - Branching Semantic-Delta Engine
Dual-process cognitive system with semantic deltas, branch trees, master tree, index, ledger
"""
import time
import uuid
from collections import defaultdict, deque
from typing import List, Dict, Any

class Delta:
    def __init__(self, payload: Any, delta_type: str='update', meta: Dict=None):
        self.id = str(uuid.uuid4())
        self.payload = payload
        self.type = delta_type
        self.meta = meta or {}
        self.timestamp = time.time()

class Branch:
    def __init__(self, branch_id: str, parent_id: str=None):
        self.id = branch_id
        self.parent_id = parent_id
        self.deltas = []
        self.stability = 1.0
        self.utility = 0.5
        self.state = 'Active'  # Active, Collapsing, Collapsed, Pruned, Bloomed
        self.created = time.time()

    def add_delta(self, delta: Delta):
        self.deltas.append(delta)
        # Update stability/utility heuristics
        self.utility = min(1.0, self.utility + 0.05)

    def score(self):
        return self.stability * self.utility * (1.0 + len(self.deltas)*0.01)

class MasterTree:
    def __init__(self):
        self.nodes = {}
        self.version = 0

    def integrate(self, branch: Branch):
        self.version += 1
        # Merge deltas into master
        for d in branch.deltas:
            self.nodes[d.id] = d
        return self.version

class Index:
    def __init__(self):
        self.store = []
    
    def add(self, delta: Delta):
        self.store.append(delta)
    
    def similarity_search(self, query: str, top_k=5):
        # Simple keyword similarity
        qwords = set(query.lower().split())
        scored = []
        for d in self.store:
            payload_str = str(d.payload).lower()
            words = set(payload_str.split())
            overlap = len(qwords & words)
            if overlap>0:
                scored.append((overlap, d))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in scored[:top_k]]

class Ledger:
    def __init__(self):
        self.events = deque(maxlen=100000)
    
    def log(self, event_type: str, data: Dict):
        self.events.append({
            'ts': time.time(),
            'type': event_type,
            'data': data
        })

class BranchManager:
    def __init__(self):
        self.branches = {}
        self.master = MasterTree()
        self.index = Index()
        self.ledger = Ledger()
        self.root_id = 'master'
        self.branches[self.root_id] = Branch(self.root_id)

    def skimming(self, delta: Delta) -> bool:
        # System 1 attention: filter low-value noise
        payload_len = len(str(delta.payload))
        if payload_len < 3:
            return False
        return True

    def bloom(self, delta: Delta) -> List[Delta]:
        # Generate semantic variants
        variants = []
        base = str(delta.payload)
        # Simple synthetic variants
        variants.append(Delta(f"variant_of_{base}", 'bloom', {'parent': delta.id}))
        return variants

    def semantic_route(self, delta: Delta):
        # Route to most similar branch
        best_branch = self.root_id
        best_score = -1
        for bid, branch in self.branches.items():
            # Simple heuristic: branch with most deltas
            score = branch.score()
            if score > best_score:
                best_score = score
                best_branch = bid
        return best_branch

    def process_delta(self, delta: Delta):
        # Skimming
        if not self.skimming(delta):
            self.ledger.log('skim_reject', {'delta_id': delta.id})
            return
        # Index store
        self.index.add(delta)
        # Bloom
        variants = self.bloom(delta)
        for v in variants:
            self.index.add(v)
        # Route
        branch_id = self.semantic_route(delta)
        branch = self.branches.get(branch_id)
        if not branch:
            branch = Branch(branch_id)
            self.branches[branch_id] = branch
        branch.add_delta(delta)
        self.ledger.log('delta_routed', {'delta_id': delta.id, 'branch_id': branch_id})
        # Predictive collapse check
        if branch.utility > 0.8 or len(branch.deltas) > 10:
            self.collapse_branch(branch_id)

    def collapse_branch(self, branch_id: str):
        branch = self.branches.get(branch_id)
        if not branch or branch.state != 'Active':
            return
        branch.state = 'Collapsing'
        self.master.integrate(branch)
        branch.state = 'Collapsed'
        self.ledger.log('collapse', {'branch_id': branch_id, 'deltas': len(branch.deltas)})
        # Reinforcement learning: reward stability
        branch.stability = min(1.0, branch.stability + 0.1)

    def spawn_branch(self, parent_id: str) -> str:
        new_id = str(uuid.uuid4())
        branch = Branch(new_id, parent_id)
        self.branches[new_id] = branch
        self.ledger.log('branch_spawn', {'new_id': new_id, 'parent': parent_id})
        return new_id

    def merge_clusters(self):
        # Merge branches with high semantic similarity
        merged = 0
        ids = list(self.branches.keys())
        for i in range(len(ids)):
            for j in range(i+1, len(ids)):
                # Simplified merge heuristic
                if self.branches[ids[i]].utility > 0.7 and self.branches[ids[j]].utility > 0.7:
                    # Merge j into i
                    b_i = self.branches[ids[i]]
                    b_j = self.branches[ids[j]]
                    for d in b_j.deltas:
                        b_i.add_delta(d)
                    self.ledger.log('merge', {'from': ids[j], 'into': ids[i]})
                    del self.branches[ids[j]]
                    merged += 1
        return merged

# Singleton
_syntheticmind = BranchManager()

def get_syntheticmind():
    return _syntheticmind
