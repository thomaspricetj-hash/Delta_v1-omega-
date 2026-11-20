"""
DaxMemoryCore - high-level symbolic memory
Facts, entities, relations, graph, ledger, index
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import json
from pathlib import Path

@dataclass
class Fact:
    id: str
    content: str
    source: Optional[str] = None
    confidence: float = 1.0

@dataclass
class Entity:
    id: str
    name: str
    attributes: Dict[str,str] = field(default_factory=dict)

@dataclass
class Relation:
    subject: str
    predicate: str
    obj: str
    confidence: float = 1.0

class DaxMemoryCore:
    def __init__(self, persist_path: Optional[Path]=None):
        self.facts: Dict[str, Fact] = {}
        self.entities: Dict[str, Entity] = {}
        self.relations: List[Relation] = []
        self.ledger: List[Dict] = []
        self.index: Dict[str, List[str]] = {}
        self.persist_path = persist_path

    def add_fact(self, fact: Fact):
        self.facts[fact.id] = fact
        self.ledger.append({"op":"add_fact","id":fact.id,"ts":__import__('time').time()})
        self._index_fact(fact)

    def add_entity(self, entity: Entity):
        self.entities[entity.id] = entity
        self.ledger.append({"op":"add_entity","id":entity.id,"ts":__import__('time').time()})

    def add_relation(self, rel: Relation):
        self.relations.append(rel)
        self.ledger.append({"op":"add_relation","subject":rel.subject,"ts":__import__('time').time()})

    def _index_fact(self, fact: Fact):
        tokens = fact.content.lower().split()
        for tok in tokens:
            self.index.setdefault(tok, []).append(fact.id)

    def recall_symbolic(self, query: str, top_k: int=10) -> List[Fact]:
        tokens = query.lower().split()
        scores = {}
        for tok in tokens:
            for fid in self.index.get(tok, []):
                scores[fid] = scores.get(fid,0)+1
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [self.facts[fid] for fid,_ in ranked if fid in self.facts]

    def save(self):
        if not self.persist_path:
            return
        data = {
            "facts": {k: vars(v) for k,v in self.facts.items()},
            "entities": {k: vars(v) for k,v in self.entities.items()},
            "relations": [vars(r) for r in self.relations],
            "ledger": self.ledger
        }
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        # Compress with BitDrop v2
        try:
            import sys
            from pathlib import Path as P
            sys.path.insert(0, str(P(__file__).resolve().parent.parent.parent))
            from bitdrop_v2_integration import compress_data
            raw_bytes = json.dumps(data).encode('utf-8')
            comp_info = compress_data(raw_bytes)
            bd_path = self.persist_path.with_suffix(self.persist_path.suffix + '.bd2')
            with open(bd_path, 'w') as f:
                json.dump({"compressed": True, "info": comp_info, "raw_len": len(raw_bytes)}, f)
        except Exception:
            pass
        with open(self.persist_path, "w") as f:
            json.dump(data, f)

    def load(self):
        if not self.persist_path or not self.persist_path.exists():
            return
        with open(self.persist_path) as f:
            data = json.load(f)
        self.facts = {k: Fact(**v) for k,v in data.get("facts",{}).items()}
        self.entities = {k: Entity(**v) for k,v in data.get("entities",{}).items()}
        self.relations = [Relation(**r) for r in data.get("relations",[])]
        self.ledger = data.get("ledger",[])
        # rebuild index
        self.index = {}
        for f in self.facts.values():
            self._index_fact(f)
