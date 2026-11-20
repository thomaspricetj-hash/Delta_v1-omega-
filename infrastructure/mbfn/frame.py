"""
MBFN Frame definition - production grade
"""
from dataclasses import dataclass, field
from typing import Set, Dict, List
import uuid

@dataclass
class Frame:
    id: str
    label: str
    layer: str
    state: int = 0  # -1 oppose, 0 inactive, 1 support
    strength: float = 0.0
    connections: Set[str] = field(default_factory=set)
    metadata: Dict = field(default_factory=dict)

    def activate(self, evidence: float):
        self.state = 1 if evidence > 0 else -1 if evidence < 0 else 0
        self.strength = abs(evidence)

    def vote(self, candidate_id: str, weight: float = 1.0) -> float:
        if self.state == 0:
            return 0.0
        return self.strength * weight * (1 if self.state > 0 else -1)

    def to_dict(self):
        return {
            "id": self.id,
            "label": self.label,
            "layer": self.layer,
            "state": self.state,
            "strength": self.strength,
            "connections": list(self.connections),
            "metadata": self.metadata
        }

    @classmethod
    def create(cls, label: str, layer: str, metadata=None):
        fid = str(uuid.uuid4())
        return cls(id=fid, label=label, layer=layer, metadata=metadata or {})
