"""
KV-Webb Core – WebNode, WebEdge, KvWeb
Python reference implementation of the diverging-semantic memory engine.
"""
import time
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple

class WebNode:
    def __init__(self, node_id: str, tokens: List[str], label: str = None, score: float = 1.0):
        self.id = node_id
        self.tokens = tokens
        self.label = label
        self.score = score
        self.created_at = time.time()
        self.last_access = self.created_at
        self.stability = 1.0
        self.branch_id = None

    def to_dict(self):
        return {
            "id": self.id,
            "tokens": self.tokens,
            "label": self.label,
            "score": self.score,
            "created_at": self.created_at,
            "last_access": self.last_access,
            "stability": self.stability,
            "branch_id": self.branch_id
        }

class WebEdge:
    def __init__(self, src: str, dst: str, weight: float = 1.0, kind: str = "Semantic"):
        self.src = src
        self.dst = dst
        self.weight = weight
        self.kind = kind
        self.last_seen = time.time()

    def to_dict(self):
        return {"src": self.src, "dst": self.dst, "weight": self.weight, "kind": self.kind, "last_seen": self.last_seen}

class KvWeb:
    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self.nodes: Dict[str, WebNode] = {}
        self.edges: Dict[Tuple[str,str], WebEdge] = {}
        self.token_index: Dict[str, Set[str]] = defaultdict(set)
        self._load()

    @staticmethod
    def _default_path():
        return Path("C:/Users/thoma/.lmstudio/apps/bionic/projects/d49037d8-47f4-5808-9028-c707de117f8f/workspace/TRANSLATION_OS_V1/runtime/logs/kv_webb.json")

    def _load(self):
        if self.storage_path.exists():
            data = json.loads(self.storage_path.read_text())
            for n in data.get("nodes", []):
                node = WebNode(**{k:v for k,v in n.items() if k in ["id","tokens","label","score"]})
                node.created_at = n.get("created_at", time.time())
                node.last_access = n.get("last_access", node.created_at)
                node.stability = n.get("stability",1.0)
                node.branch_id = n.get("branch_id")
                self.nodes[node.id] = node
            for e in data.get("edges", []):
                edge = WebEdge(e["src"], e["dst"], e.get("weight",1.0), e.get("kind","Semantic"))
                edge.last_seen = e.get("last_seen", time.time())
                self.edges[(edge.src, edge.dst)] = edge
            # rebuild token index
            for nid, node in self.nodes.items():
                for tok in node.tokens:
                    self.token_index[tok].add(nid)

    def _save(self):
        data = {
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": [e.to_dict() for e in self.edges.values()]
        }
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.storage_path.write_text(json.dumps(data))

    def create_node(self, tokens: List[str], label: str = None, score: float = 1.0):
        import uuid
        nid = str(uuid.uuid4())
        node = WebNode(nid, tokens, label, score)
        self.nodes[nid] = node
        for tok in tokens:
            self.token_index[tok].add(nid)
        self._save()
        return node

    def add_edge(self, src_id: str, dst_id: str, weight: float = 1.0, kind: str = "Semantic"):
        key = (src_id, dst_id)
        if key in self.edges:
            self.edges[key].weight += weight
            self.edges[key].last_seen = time.time()
        else:
            self.edges[key] = WebEdge(src_id, dst_id, weight, kind)
        self._save()

    def reinforce(self, node_ids: List[str], reward: float = 1.0):
        for nid in node_ids:
            if nid in self.nodes:
                node = self.nodes[nid]
                node.score += reward
                node.stability = min(1.0, node.stability + 0.05)
                node.last_access = time.time()
        self._save()

    def apply_drift(self, decay: float = 0.001):
        now = time.time()
        for node in self.nodes.values():
            age = now - node.last_access
            node.score *= (1.0 - decay * age / 86400.0)
            node.stability *= (1.0 - decay * 0.5 * age / 86400.0)
        self._save()

    def prune(self, score_threshold: float = 0.01):
        to_remove = [nid for nid, n in self.nodes.items() if n.score < score_threshold]
        for nid in to_remove:
            # remove edges
            keys = [k for k in self.edges if k[0]==nid or k[1]==nid]
            for k in keys:
                del self.edges[k]
            # remove from token index
            node = self.nodes.pop(nid)
            for tok in node.tokens:
                s = self.token_index.get(tok)
                if s:
                    s.discard(nid)
        self._save()
        return len(to_remove)

    def retrieve_region(self, query_tokens: List[str], top_k: int = 20):
        # BFS-like relevance via token overlap and PageRank-ish scoring
        candidates = set()
        for tok in query_tokens:
            candidates.update(self.token_index.get(tok, set()))
        scored = []
        for nid in candidates:
            node = self.nodes.get(nid)
            if not node:
                continue
            overlap = len(set(query_tokens) & set(node.tokens))
            score = node.score * node.stability * (1 + overlap)
            scored.append((score, nid, node))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [n for _,_,n in scored[:top_k]]
