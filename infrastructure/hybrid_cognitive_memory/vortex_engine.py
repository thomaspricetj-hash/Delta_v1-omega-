"""
Vortex Engine - geometric delta routing
Rings, radii, subsystem angles, collapse
"""
from dataclasses import dataclass, field
from typing import Dict, List
import math

@dataclass
class VortexDelta:
    id: str
    radius: float
    angle: float  # radians, subsystem angle
    subsystem: str
    value: float
    timestamp: float
    confidence: float = 1.0

@dataclass
class Ring:
    radius_min: float
    radius_max: float
    deltas: List[VortexDelta] = field(default_factory=list)

    def add(self, delta: VortexDelta):
        if self.radius_min <= delta.radius <= self.radius_max:
            self.deltas.append(delta)
            return True
        return False

class VortexEngine:
    SUBSYSTEMS = ["Memory","Knowledge","Linguistics","Compression","Routing"]

    def __init__(self, num_rings=5, max_radius=10.0):
        self.rings: List[Ring] = []
        step = max_radius / num_rings
        for i in range(num_rings):
            self.rings.append(Ring(i*step, (i+1)*step))

    def route_delta(self, subsystem: str, radius: float, value: float, confidence: float=1.0) -> VortexDelta:
        idx = self.SUBSYSTEMS.index(subsystem) if subsystem in self.SUBSYSTEMS else 0
        angle = (idx / len(self.SUBSYSTEMS)) * 2 * math.pi
        delta = VortexDelta(
            id=f"{subsystem}_{radius:.2f}_{value:.3f}",
            radius=radius,
            angle=angle,
            subsystem=subsystem,
            value=value,
            timestamp=__import__('time').time(),
            confidence=confidence
        )
        for ring in self.rings:
            if ring.add(delta):
                break
        return delta

    def collapse_up_to_radius(self, radius_limit: float, memory_frame):
        collapsed = []
        for ring in self.rings:
            if ring.radius_max > radius_limit:
                continue
            for delta in ring.deltas:
                # Map to frame grid coordinates
                x = int(delta.radius)
                y = int(math.degrees(delta.angle)) % 10
                # Write to appropriate slice based on subsystem
                slice_map = {
                    "Memory": "semantic",
                    "Knowledge": "relational",
                    "Linguistics": "temporal",
                    "Compression": "spatial",
                    "Routing": "semantic"
                }
                slice_name = slice_map.get(delta.subsystem, "semantic")
                memory_frame.write(slice_name, x, y, delta.value, delta.confidence)
                collapsed.append(delta.id)
        # Clear processed deltas
        for ring in self.rings:
            ring.deltas = [d for d in ring.deltas if d.radius > radius_limit]
        return collapsed

    def get_ring_summary(self):
        return [{"radius_min": r.radius_min, "radius_max": r.radius_max, "delta_count": len(r.deltas)} for r in self.rings]
