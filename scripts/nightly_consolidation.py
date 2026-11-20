"""
Cross-layer consolidation nightly
Run master_map_integrator + planner + graph pruning
"""
import sys
from pathlib import Path
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "infrastructure"))
sys.path.insert(0, str(BASE / "scripts"))

def prune_graph(state, weight_threshold=0.1):
    # Prune low-weight edges/clusters
    pruned = 0
    layers = state.get("layers", {})
    # layers is dict {id: layer_dict}
    for layer_id, layer in layers.items():
        edges = layer.get("edges", [])
        original = len(edges)
        layer["edges"] = [e for e in edges if e.get("weight",1.0) >= weight_threshold]
        pruned += original - len(layer["edges"])
    return pruned

def run():
    # Integrate master map
    from master_map_integrator import integrate_all
    master = integrate_all()
    # Prune low-weight edges
    pruned = prune_graph(master, weight_threshold=0.15)
    # Update planner
    from planner import update_progress
    roadmap, prog = update_progress()
    return {"layers": len(master.get("layers",[])), "pruned_edges": pruned, "roadmap": roadmap}

if __name__ == "__main__":
    print(run())
