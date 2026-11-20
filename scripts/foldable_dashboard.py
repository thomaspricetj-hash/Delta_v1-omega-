"""
Foldable Dashboard CLI
Shows live foldable layer sizes, grid links and lattice stats
"""
import sys
from pathlib import Path
BASE = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(BASE / "infrastructure"))
from foldable_map.foldable_map import _foldable_map
sys.path.insert(0, str(BASE / "infrastructure" / "weave"))
from lattice_weave_interference import get_lattice_summary

def print_dashboard():
    layer_sizes = {i: len(layer) for i, layer in enumerate(_foldable_map.layers)}
    total_items = sum(layer_sizes.values())
    ledger_len = len(_foldable_map.ledger)
    lattice = get_lattice_summary()
    # Abbreviation map stats
    try:
        sys.path.insert(0, str(BASE / "infrastructure" / "abbreviation_map"))
        from abbrev_map import get_map
        from bitdrop_persistence import load_json_bitdrop
        fwd, rev = get_map()
        counter_path = BASE / "runtime" / "logs" / "abbrev_counter.json"
        counter_data = load_json_bitdrop(counter_path) or {}
        abbrev_stats = {
            "map_size": len(fwd),
            "counter_entries": len(counter_data),
            "top_phrases": sorted(counter_data.items(), key=lambda x: x[1], reverse=True)[:5]
        }
    except Exception:
        abbrev_stats = {"map_size": 0, "counter_entries": 0, "top_phrases": []}
    print("=== Foldable Multilayer Dashboard ===")
    print(f"Layers: {len(_foldable_map.layers)}")
    for i, size in layer_sizes.items():
        print(f"  Layer {i}: {size} items")
    print(f"Total items: {total_items}")
    print(f"Ledger entries: {ledger_len}")
    print(f"Access counts tracked: {len(_foldable_map.access_counts)}")
    print("")
    print("=== Lattice Weave ===")
    print(f"Nodes: {lattice.get('nodes',0)}")
    print(f"Links: {lattice.get('links',0)}")
    print(f"History entries: {lattice.get('history_len',0)}")
    print("")
    print("=== Grid Innerconnects ===")
    # Count grid nodes in layer 0
    grid_nodes = [k for k in _foldable_map.layers[0] if k.startswith('grid:')]
    lattice_nodes = [k for k in _foldable_map.layers[1] if k.startswith('lattice:')]
    print(f"Grid nodes in layer 0: {len(grid_nodes)}")
    print(f"Lattice nodes in layer 1: {len(lattice_nodes)}")
    if grid_nodes:
        print("Recent grid nodes:")
        for k in grid_nodes[-5:]:
            print(f"  {k}")
    print("")
    print("=== Abbreviation Map ===")
    print(f"Map size: {abbrev_stats.get('map_size',0)} entries")
    print(f"Counter entries: {abbrev_stats.get('counter_entries',0)}")
    top = abbrev_stats.get('top_phrases',[])
    if top:
        print("Top phrases:")
        for phrase, cnt in top:
            print(f"  {phrase!r}: {cnt}")
    print("")
    print("Dashboard ready.")

if __name__ == "__main__":
    print_dashboard()
