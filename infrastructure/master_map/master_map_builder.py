"""
Master Map Builder - Multi-Layered
Aggregates all maps into layered holistic view.
Layers:
  L0_raw      - raw map files
  L1_processed - categorized + meaning enriched
  L2_zones    - memory zoning
  L3_helpers  - helpers & micro helpers
  L4_fiber    - fiber harmony correlation
"""
import json
import sys
from pathlib import Path
from datetime import datetime, timezone

BASE = Path(__file__).resolve().parent.parent.parent.parent
RUNTIME_LOGS = BASE / "runtime" / "logs"
MEMORY_ZONES = RUNTIME_LOGS / "memory_zones"
FIBER_STATE = BASE / "infrastructure" / "fiber_harmony" / "fiber_state.json"
HELPERS_DIR = BASE / "infrastructure" / "helpers"

MASTER_MAP_PATH = RUNTIME_LOGS / "master_map.json"
MASTER_INDEX_PATH = RUNTIME_LOGS / "master_index.json"

MAPS = {
    "semantic_map": RUNTIME_LOGS / "semantic_map.json",
    "semantic_index": RUNTIME_LOGS / "semantic_index.json",
    "semantic_graph": RUNTIME_LOGS / "semantic_graph.json",
    "pattern_map": RUNTIME_LOGS / "pattern_map.json",
    "next_word_map": RUNTIME_LOGS / "next_word_map.json",
    "stat_word_relations": RUNTIME_LOGS / "stat_word_relations.json",
    "word_meaning_map": RUNTIME_LOGS / "word_meaning_map.json",
    "predictive_index": RUNTIME_LOGS / "predictive_index.json",
}

def _load(path):
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}

def _save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")

def build_master_map():
    # Layer 0: Raw maps
    layer_raw = {}
    for name, path in MAPS.items():
        data = _load(path)
        if data:
            layer_raw[name] = data
    
    # Layer 1: Processed / categorized with meaning enrichment
    processed = {}
    for cat in ["semantic_facts", "semantic_words", "semantic_connections"]:
        data = _load(RUNTIME_LOGS / f"{cat}.json")
        if data:
            processed[cat] = data
    
    # Layer 2: Memory zones
    zones = {}
    if MEMORY_ZONES.exists():
        for zone_file in MEMORY_ZONES.glob("*.json"):
            if zone_file.name == "zone_metadata.json":
                continue
            zones[zone_file.stem] = _load(zone_file)
    
    # Layer 3: Helpers & Micro Helpers
    helpers_meta = {}
    micro_helpers_meta = {}
    try:
        sys.path.insert(0, str(HELPERS_DIR))
        import helpers as helpers_module
        helpers_meta["module"] = "helpers"
        helpers_meta["functions"] = [n for n in dir(helpers_module) if not n.startswith("_")]
        import micro_helpers as mh_module
        micro_helpers_meta["module"] = "micro_helpers"
        micro_helpers_meta["functions"] = [n for n in dir(mh_module) if not n.startswith("_")]
    except Exception:
        helpers_meta["error"] = "load failed"
        micro_helpers_meta["error"] = "load failed"
    
    # Layer 4: Fiber harmony
    fiber = _load(FIBER_STATE) if FIBER_STATE.exists() else {}
    
    master = {
        "metadata": {
            "built_at": datetime.now(timezone.utc).isoformat(),
            "layers": ["L0_raw", "L1_processed", "L2_zones", "L3_helpers", "L4_fiber"]
        },
        "layers": {
            "L0_raw": layer_raw,
            "L1_processed": processed,
            "L2_zones": zones,
            "L3_helpers": {
                "helpers": helpers_meta,
                "micro_helpers": micro_helpers_meta
            },
            "L4_fiber": fiber
        },
        "summary": {}
    }
    
    # Summary
    summary = {
        "L0_raw": {k: len(v) if isinstance(v, dict) else 0 for k, v in layer_raw.items()},
        "L1_processed": {k: len(v) if isinstance(v, dict) else 0 for k, v in processed.items()},
        "L2_zones": {k: len(v) if isinstance(v, dict) else 0 for k, v in zones.items()},
        "L3_helpers": {
            "helpers": len(helpers_meta.get("functions", [])),
            "micro_helpers": len(micro_helpers_meta.get("functions", []))
        },
        "L4_fiber": len(fiber) if isinstance(fiber, dict) else 0
    }
    master["summary"] = summary
    
    _save_json(MASTER_MAP_PATH, master)
    
    # Multi-layered index
    index = {
        "metadata": master["metadata"],
        "summary": summary,
        "layers": {
            "L0_raw": list(layer_raw.keys()),
            "L1_processed": list(processed.keys()),
            "L2_zones": list(zones.keys()),
            "L3_helpers": {
                "helpers": helpers_meta.get("functions", []),
                "micro_helpers": micro_helpers_meta.get("functions", [])
            },
            "L4_fiber": list(fiber.keys()) if isinstance(fiber, dict) else []
        }
    }
    _save_json(MASTER_INDEX_PATH, index)
    
    return master

def get_master_snapshot():
    if MASTER_MAP_PATH.exists():
        return _load(MASTER_MAP_PATH)
    return None

def query_master(query):
    master = get_master_snapshot()
    if not master:
        build_master_map()
        master = get_master_snapshot()
    
    results = {"query": query, "hits": {}}
    # Search across layers
    for layer_name, layer_data in master.get("layers", {}).items():
        if layer_name == "L3_helpers":
            continue
        if isinstance(layer_data, dict):
            hits = {}
            for k, v in layer_data.items():
                if isinstance(v, dict):
                    matched_keys = [kk for kk in v.keys() if query.lower() in kk.lower()]
                    if matched_keys:
                        hits[k] = matched_keys[:10]
            if hits:
                results["hits"][layer_name] = hits
    return results

if __name__ == "__main__":
    m = build_master_map()
    print(f"Master map built. Layers: {list(m['layers'].keys())}")
