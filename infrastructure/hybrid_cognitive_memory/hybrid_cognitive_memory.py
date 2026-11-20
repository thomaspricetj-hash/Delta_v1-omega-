"""
Hybrid Cognitive Memory System
MemoryFrame + Vortex Rings + DaxMemoryCore
Unified interface
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from memory_frame import MemoryFrame
from vortex_engine import VortexEngine
from dax_memory_core import DaxMemoryCore, Fact
from pathlib import Path
import time

class DaxVortexMaster:
    def __init__(self, persist_path: str = None):
        self.frame = MemoryFrame()
        self.vortex = VortexEngine(num_rings=5, max_radius=10.0)
        core_path = Path(persist_path) if persist_path else None
        self.core = DaxMemoryCore(persist_path=core_path)
        if core_path and core_path.exists():
            self.core.load()

    def write_fact(self, fact: Fact):
        self.core.add_fact(fact)
        # Route to vortex as delta
        radius = min(10.0, max(0.1, fact.confidence * 10))
        subsystem = "Knowledge" if "is" in fact.content.lower() else "Memory"
        self.vortex.route_delta(subsystem, radius, fact.confidence, fact.confidence)
        # Also write to frame directly
        x = hash(fact.id) % 100
        y = hash(fact.content) % 100
        self.frame.write("semantic", x, y, fact.confidence, fact.confidence)

    def collapse_up_to_radius(self, radius_limit: float):
        collapsed = self.vortex.collapse_up_to_radius(radius_limit, self.frame)
        self.frame.decay_all()
        return collapsed

    def recall_blended(self, query: str, top_k: int = 10):
        # Symbolic recall
        symbolic_facts = self.core.recall_symbolic(query, top_k)
        # Frame recall via index heuristic
        tokens = query.lower().split()
        frame_hits = []
        for tok in tokens:
            x = hash(tok) % 100
            y = hash(query) % 100
            cell = self.frame.read("semantic", x, y)
            if cell.confidence > 0:
                frame_hits.append((cell.value, cell.confidence))
        # Blend scores
        results = []
        for f in symbolic_facts:
            score = f.confidence
            # boost if frame has recent activity
            if frame_hits:
                avg_conf = sum(c for _,c in frame_hits)/len(frame_hits)
                score += avg_conf * 0.3
            results.append({
                "fact_id": f.id,
                "content": f.content,
                "confidence": round(score,3),
                "source": "blended"
            })
        # Sort
        results.sort(key=lambda r: r["confidence"], reverse=True)
        return results[:top_k]

    def save(self):
        self.core.save()

# Convenience factory
def with_default_applier(persist_path=None):
    return DaxVortexMaster(persist_path=persist_path)
