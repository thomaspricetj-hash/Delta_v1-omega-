"""
Structural Learning Engine integration bridge for Translation OS v1
Production grade Python implementation of Semantic Graph, Structural Learner, DAX Journaling
"""
import json
import uuid
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Set, Tuple
import threading

@dataclass
class EpisodeTrace:
    tokens: List[str]
    activations: List[List[float]]
    tags: List[str]
    time: float = 0.0
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

@dataclass
class Concept:
    id: str
    label: str
    embedding: List[float] = field(default_factory=list)
    degree: int = 0
    metadata: Dict = field(default_factory=dict)

@dataclass
class Relation:
    id: str
    subject_id: str
    predicate: str
    object_id: str
    weight: float = 1.0

class SemanticGraph:
    def __init__(self):
        self.concepts: Dict[str, Concept] = {}
        self.relations: Dict[str, Relation] = {}
        self.label_to_id: Dict[str, str] = {}
        self.lock = threading.Lock()

    def get_or_create_concept(self, label: str) -> str:
        with self.lock:
            if label in self.label_to_id:
                cid = self.label_to_id[label]
                self.concepts[cid].degree += 1
                return cid
            cid = str(uuid.uuid4())
            concept = Concept(id=cid, label=label)
            self.concepts[cid] = concept
            self.label_to_id[label] = cid
            return cid

    def add_relation(self, subject_label: str, predicate: str, object_label: str, weight: float = 1.0):
        s_id = self.get_or_create_concept(subject_label)
        o_id = self.get_or_create_concept(object_label)
        rid = str(uuid.uuid4())
        rel = Relation(id=rid, subject_id=s_id, predicate=predicate, object_id=o_id, weight=weight)
        with self.lock:
            self.relations[rid] = rel

    def shortest_path(self, start_label: str, end_label: str) -> List[str]:
        # BFS
        start_id = self.label_to_id.get(start_label)
        end_id = self.label_to_id.get(end_label)
        if not start_id or not end_id:
            return []
        from collections import deque
        queue = deque([(start_id, [start_id])])
        visited = {start_id}
        adj = {}
        for r in self.relations.values():
            adj.setdefault(r.subject_id, []).append((r.object_id, r.predicate))
            adj.setdefault(r.object_id, []).append((r.subject_id, r.predicate))
        while queue:
            node, path = queue.popleft()
            if node == end_id:
                return [self.concepts[n].label for n in path]
            for nb, _ in adj.get(node, []):
                if nb not in visited:
                    visited.add(nb)
                    queue.append((nb, path + [nb]))
        return []

    def to_dict(self):
        return {
            'concepts': [asdict(c) for c in self.concepts.values()],
            'relations': [asdict(r) for r in self.relations.values()]
        }

    @classmethod
    def from_dict(cls, data):
        g = cls()
        for c in data.get('concepts', []):
            concept = Concept(**c)
            g.concepts[concept.id] = concept
            g.label_to_id[concept.label] = concept.id
        for r in data.get('relations', []):
            rel = Relation(**r)
            g.relations[rel.id] = rel
        return g

class DAXJournal:
    def __init__(self, journal_dir: Path):
        self.dir = Path(journal_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.dir / "journal.jsonl"
        # DAX Registry for compressed journaling
        try:
            from infrastructure.dax_registry.dax_registry import DAXRegistry
            self.dax = DAXRegistry(self.dir / 'dax')
            self.dax_enabled = True
        except Exception:
            self.dax = None
            self.dax_enabled = False

    def append_diff(self, diff: Dict):
        line = json.dumps(diff)
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(line + '\n')
        # Store compressed delta via DAX Registry
        if self.dax_enabled:
            try:
                delta_id = f"diff_{diff.get('episode_id','')}_{self.dir.name}"
                self.dax.add_delta(delta_id, line.encode('utf-8'), is_dax=True)
            except Exception:
                pass

    def replay(self) -> List[Dict]:
        diffs = []
        if self.log_path.exists():
            with open(self.log_path, 'r', encoding='utf-8') as f:
                for line in f:
                    diffs.append(json.loads(line))
        return diffs

class StructuralLearner:
    def __init__(self, journal_dir: Path = Path("runtime/structural/journal")):
        self.graph = SemanticGraph()
        self.journal = DAXJournal(journal_dir)
        self.version = 0

    def learn_from(self, episode: EpisodeTrace):
        # Concept creation from tokens
        for token in episode.tokens:
            self.graph.get_or_create_concept(token)
        # Simple relation induction: adjacent tokens -> SimilarTo / Implies
        for i in range(len(episode.tokens)-1):
            s = episode.tokens[i]
            o = episode.tokens[i+1]
            self.graph.add_relation(s, "SimilarTo", o, weight=0.5)
        # Update embeddings placeholder
        # Journal diff
        diff = {
            'episode_id': episode.id,
            'tokens': episode.tokens,
            'tags': episode.tags,
            'time': episode.time,
            'version': self.version
        }
        self.journal.append_diff(diff)
        self.version += 1

    def save_state(self, path: Path):
        data = self.graph.to_dict()
        data['version'] = self.version
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f)

    def load_state(self, path: Path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.graph = SemanticGraph.from_dict(data)
        self.version = data.get('version', 0)

    def explain(self, concept_label: str, depth: int = 2) -> List[Dict]:
        # Simple neighborhood explanation
        cid = self.graph.label_to_id.get(concept_label)
        if not cid:
            return []
        related = []
        for r in self.graph.relations.values():
            if r.subject_id == cid or r.object_id == cid:
                related.append({
                    'predicate': r.predicate,
                    'related': self.graph.concepts[r.object_id if r.subject_id == cid else r.subject_id].label,
                    'weight': r.weight
                })
        return sorted(related, key=lambda x: x['weight'], reverse=True)[:10]

class LearnerBridge:
    def __init__(self, journal_dir: Path = Path("runtime/structural/journal")):
        self.learner = StructuralLearner(journal_dir)

    def ingest_batch(self, texts: List[str], activations: List[List[float]], tags: List[str]):
        for text, act, tag in zip(texts, activations, tags):
            tokens = text.split()
            ep = EpisodeTrace(tokens=tokens, activations=[act], tags=[tag], time=0.0)
            self.learner.learn_from(ep)

    def get_graph(self) -> SemanticGraph:
        return self.learner.graph
