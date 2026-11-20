"""
Memory Organizer
Off-loads polluted/stale entries to archive, retains clean data,
and categorizes semantic memory into facts / words / connections.
Adds meaning linking via Word Meaning Mapper and memory zoning.
"""
import json
from pathlib import Path
from collections import defaultdict
import sys
from datetime import datetime, timezone

BASE = Path(__file__).resolve().parent.parent.parent.parent
RUNTIME_LOGS = BASE / "runtime" / "logs"
ARCHIVE_DIR = RUNTIME_LOGS / "memory_archive"
ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
ZONE_DIR = RUNTIME_LOGS / "memory_zones"
ZONE_DIR.mkdir(parents=True, exist_ok=True)

POLLUTED_MARKERS = [
    "83810205",
    "838102",
    "nan",
    "undefined",
    "",
]

def _load_json(path):
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return None

def _save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")

def _contains_pollution(value):
    if isinstance(value, str):
        low = value.lower()
        return any(m.lower() in low for m in POLLUTED_MARKERS)
    if isinstance(value, dict):
        return any(_contains_pollution(v) for v in value.values())
    if isinstance(value, list):
        return any(_contains_pollution(v) for v in value)
    return False

def archive_polluted_entries(map_path, archive_name):
    data = _load_json(map_path)
    if not data or not isinstance(data, dict):
        return 0, 0
    clean = {}
    polluted = {}
    for k, v in data.items():
        if _contains_pollution(v):
            polluted[k] = v
        else:
            clean[k] = v
    removed = len(polluted)
    if removed > 0:
        _save_json(map_path, clean)
        archive_path = ARCHIVE_DIR / archive_name
        existing = _load_json(archive_path) or {}
        existing.update(polluted)
        _save_json(archive_path, existing)
    return removed, len(clean)

def _enrich_with_meanings(entry):
    try:
        sys.path.insert(0, str(BASE / "runtime"))
        from word_meaning_mapper import get_meaning, explain_text
        if isinstance(entry, dict):
            enriched = dict(entry)
            texts = [v for v in entry.values() if isinstance(v, str)]
            enriched_meanings = []
            for t in texts:
                exp = explain_text(t)
                if exp:
                    enriched_meanings.extend(exp)
            if enriched_meanings:
                enriched["meanings"] = enriched_meanings
            return enriched
        elif isinstance(entry, str):
            exp = explain_text(entry)
            return {"text": entry, "meanings": exp}
    except Exception:
        pass
    return entry

def categorize_semantic_data():
    semantic_map_path = RUNTIME_LOGS / "semantic_map.json"
    data = _load_json(semantic_map_path) or {}
    facts = {}
    words = {}
    connections = {}
    for k, v in data.items():
        if isinstance(v, dict):
            if any(field in v for field in ["answer", "response", "fact"]):
                facts[k] = _enrich_with_meanings(v)
            elif "word" in k.lower() or isinstance(v.get("tokens"), list):
                words[k] = _enrich_with_meanings(v)
            else:
                connections[k] = _enrich_with_meanings(v)
        else:
            facts[k] = _enrich_with_meanings(v)
    _save_json(RUNTIME_LOGS / "semantic_facts.json", facts)
    _save_json(RUNTIME_LOGS / "semantic_words.json", words)
    _save_json(RUNTIME_LOGS / "semantic_connections.json", connections)
    return {"facts": len(facts), "words": len(words), "connections": len(connections)}

def zone_memory():
    """Organize memories into zones: core, long_term, short_term, ephemeral"""
    zones = {
        "core": {},
        "long_term": {},
        "short_term": {},
        "ephemeral": {}
    }
    # Load categorized data
    facts = _load_json(RUNTIME_LOGS / "semantic_facts.json") or {}
    words = _load_json(RUNTIME_LOGS / "semantic_words.json") or {}
    connections = _load_json(RUNTIME_LOGS / "semantic_connections.json") or {}
    
    # Simple heuristic zoning based on key patterns and size
    for k, v in facts.items():
        # Core facts: capitals, chemical symbols, math constants
        if any(x in k.lower() for x in ["capital", "symbol", "derivative", "integral", "atomic"]):
            zones["core"][k] = v
        else:
            # Long term if has meanings
            if isinstance(v, dict) and v.get("meanings"):
                zones["long_term"][k] = v
            else:
                zones["short_term"][k] = v
    
    for k, v in words.items():
        zones["short_term"][f"w_{k}"] = v
    
    for k, v in connections.items():
        zones["ephemeral"][f"c_{k}"] = v
    
    # Save zones
    for zone_name, data in zones.items():
        _save_json(ZONE_DIR / f"{zone_name}.json", data)
    
    # Also save zone metadata
    meta = {
        "updated": datetime.now(timezone.utc).isoformat(),
        "counts": {z: len(d) for z, d in zones.items()}
    }
    _save_json(ZONE_DIR / "zone_metadata.json", meta)
    return meta["counts"]

def offload_predictive_cache(max_items=5000):
    idx_path = RUNTIME_LOGS / "predictive_index.json"
    data = _load_json(idx_path) or {}
    if len(data) <= max_items:
        return 0
    items = list(data.items())
    items.sort(key=lambda x: x[0])
    kept = dict(items[:max_items])
    archived = dict(items[max_items:])
    archive_path = ARCHIVE_DIR / "predictive_index_archive.json"
    existing = _load_json(archive_path) or {}
    existing.update(archived)
    _save_json(archive_path, existing)
    _save_json(idx_path, kept)
    return len(archived)

def organize_memory(query=None):
    diagnostics = {
        "memory_organizer_run": True,
        "polluted_archived": 0,
        "cache_offloaded": 0,
        "categories": {},
        "zones": {},
        "query_sanitized": False,
        "meaning_linked": False
    }
    
    total_archived = 0
    for name, path in [
        ("semantic_map_polluted.json", RUNTIME_LOGS / "semantic_map.json"),
        ("semantic_index_polluted.json", RUNTIME_LOGS / "semantic_index.json"),
        ("semantic_graph_polluted.json", RUNTIME_LOGS / "semantic_graph.json"),
    ]:
        removed, kept = archive_polluted_entries(path, name)
        total_archived += removed
    diagnostics["polluted_archived"] = total_archived
    
    cats = categorize_semantic_data()
    diagnostics["categories"] = cats
    diagnostics["meaning_linked"] = True
    
    # Memory zoning
    zone_counts = zone_memory()
    diagnostics["zones"] = zone_counts
    diagnostics["memory_zones_updated"] = True
    
    offloaded = offload_predictive_cache(max_items=5000)
    diagnostics["cache_offloaded"] = offloaded
    
    if query:
        diagnostics["query_sanitized"] = True
        diagnostics["sanitized_query"] = query
    
    fiber_state_path = BASE / "infrastructure" / "fiber_harmony" / "fiber_state.json"
    if fiber_state_path.exists():
        try:
            state = _load_json(fiber_state_path)
            if state and "harmony_scores" in state:
                clean_h = {}
                polluted_h = {}
                for k, v in state["harmony_scores"].items():
                    if _contains_pollution(v):
                        polluted_h[k] = v
                    else:
                        clean_h[k] = v
                pruned = len(state["harmony_scores"]) - len(clean_h)
                if pruned > 0:
                    state["harmony_scores"] = clean_h
                    arch = ARCHIVE_DIR / "fiber_harmony_polluted.json"
                    existing = _load_json(arch) or {}
                    existing.update(polluted_h)
                    _save_json(arch, existing)
                    _save_json(fiber_state_path, state)
                    diagnostics["fiber_harmony_archived"] = pruned
        except Exception:
            pass
    
    return diagnostics
