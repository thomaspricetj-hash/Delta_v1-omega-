"""
Pattern Graph – symbolic graph of PTS patterns
"""
from dataclasses import dataclass, field
from typing import Dict, Tuple

@dataclass
class PTS:
    pattern_id: int
    tag: str
    signature: bytes = b''

import json, os

class PatternGraph:
    def __init__(self, persist_path=None):
        self.nodes: Dict[int, PTS] = {}
        self.edges: Dict[Tuple[int,int], dict] = {}
        self.freq: Dict[int,int] = {}
        self.last_prune = 0
        self.persist_path = persist_path
        if persist_path and os.path.exists(persist_path):
            self.load(persist_path)
    
    def save(self, path=None):
        p = path or self.persist_path
        if not p:
            return
        data = {
            "nodes": {str(k): {"pattern_id": v.pattern_id, "tag": v.tag, "signature": v.signature.hex()} for k,v in self.nodes.items()},
            "freq": self.freq,
            "edges": {f"{k[0]}-{k[1]}": v for k,v in self.edges.items()}
        }
        with open(p, 'w') as f:
            json.dump(data, f)
    
    def load(self, path):
        with open(path, 'r') as f:
            data = json.load(f)
        self.nodes = {}
        for k,v in data.get("nodes", {}).items():
            pts = PTS(pattern_id=v["pattern_id"], tag=v["tag"], signature=bytes.fromhex(v["signature"]))
            self.nodes[int(k)] = pts
        self.freq = data.get("freq", {})
        self.edges = {}
        for k,v in data.get("edges", {}).items():
            a,b = k.split("-")
            self.edges[(int(a), int(b))] = v
    
    def prune(self, min_freq=2):
        to_remove = [pid for pid, f in self.freq.items() if f < min_freq]
        for pid in to_remove:
            self.nodes.pop(pid, None)
            self.freq.pop(pid, None)
            # remove edges
            self.edges = {k:v for k,v in self.edges.items() if k[0]!=pid and k[1]!=pid}
        return len(to_remove)
    
    def add_pattern(self, pts: PTS):
        self.nodes[pts.pattern_id] = pts
        self.freq[pts.pattern_id] = self.freq.get(pts.pattern_id, 0) + 1
    
    def add_transition(self, src_id: int, dst_id: int, meta: dict = None):
        self.edges[(src_id, dst_id)] = meta or {}
    
    def get_stable_nodes(self, min_freq=2):
        return [pid for pid, f in self.freq.items() if f >= min_freq]

class StabilityRules:
    def __init__(self, min_freq=2, max_nodes=10000):
        self.min_freq = min_freq
        self.max_nodes = max_nodes
    
    def apply(self, pg):
        pruned = pg.prune(self.min_freq)
        # enforce max nodes
        if len(pg.nodes) > self.max_nodes:
            # drop lowest freq nodes
            sorted_nodes = sorted(pg.freq.items(), key=lambda x: x[1])
            to_drop = len(pg.nodes) - self.max_nodes
            for pid,_ in sorted_nodes[:to_drop]:
                pg.nodes.pop(pid, None)
                pg.freq.pop(pid, None)
        return pruned

class BD3DCompressor:
    def __init__(self, sim_threshold=0.95):
        self.sim_threshold = sim_threshold
        self.pattern_bank = []  # list of (centroid, pattern_id, tag)
        self.next_id = 0
    
    def _cosine_sim(self, a, b):
        import torch.nn.functional as F
        return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
    
    def compress(self, deltas: torch.Tensor):
        # deltas: [batch, seq-1, dim] – use first batch
        if deltas.numel() == 0:
            return []
        seq = deltas[0]
        pts_list = []
        for i in range(seq.shape[0]):
            vec = seq[i]
            # find nearest pattern
            best_id = None
            best_sim = -1.0
            for centroid, pid, tag in self.pattern_bank:
                sim = self._cosine_sim(vec, centroid)
                if sim > best_sim:
                    best_sim = sim
                    best_id = pid
            if best_id is not None and best_sim >= self.sim_threshold:
                pid = best_id
                tag = tag
            else:
                pid = self.next_id
                tag = "generic"
                self.pattern_bank.append((vec.clone().detach(), pid, tag))
                self.next_id += 1
            # simple signature = first 4 bytes of hash
            sig = abs(hash(vec.sum().item())) & 0xffffffff
            signature = sig.to_bytes(4, 'little', signed=False)
            pts_list.append(PTS(pattern_id=pid, tag=tag, signature=signature))
        return pts_list
