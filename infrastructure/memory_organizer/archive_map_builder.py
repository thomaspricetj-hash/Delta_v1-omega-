"""
Multilayer Archive Map Builder
Layers expand and unfold archived artefacts, link to active maps as memory extension.
"""
import json
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

BASE = Path(__file__).resolve().parent.parent.parent.parent
RUNTIME_LOGS = BASE / "runtime" / "logs"
ARCHIVE_DIR = RUNTIME_LOGS / "memory_archive"
MEMORY_ZONES = RUNTIME_LOGS / "memory_zones"

ARCHIVE_MAP_PATH = RUNTIME_LOGS / "archive_map_multilayer.json"
ARCHIVE_INDEX_PATH = RUNTIME_LOGS / "archive_index.json"

POLLUTED_MARKERS = ["83810205","838102","nan","undefined",""]

def _load_json(p):
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    except:
        return {}

def _save_json(p, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")

def _contains_pollution(v):
    if isinstance(v,str):
        low=v.lower()
        return any(m.lower() in low for m in POLLUTED_MARKERS)
    if isinstance(v,dict):
        return any(_contains_pollution(x) for x in v.values())
    if isinstance(v,list):
        return any(_contains_pollution(x) for x in v)
    return False

def collect_archive_artifacts():
    artifacts = {}
    for f in ARCHIVE_DIR.glob("*.json"):
        data = _load_json(f)
        if isinstance(data, dict):
            artifacts[f.name] = data
        elif isinstance(data, list):
            artifacts[f.name] = {str(i):v for i,v in enumerate(data)}
    return artifacts

def build_layer_l0_raw(artifacts):
    return {"layer":"L0_raw_archive","count":sum(len(v) for v in artifacts.values()),"artifacts":artifacts}

def build_layer_l1_dedup(artifacts):
    seen=set()
    dedup={}
    for name, data in artifacts.items():
        for k,v in data.items():
            key=f"{k}::{json.dumps(v,sort_keys=True)[:200]}"
            if key not in seen:
                seen.add(key)
                dedup[f"{name}::{k}"]={"key":k,"value":v,"source":name}
    return {"layer":"L1_dedup","count":len(dedup),"items":dedup}

def build_layer_l2_meaning_enriched(dedup):
    enriched={}
    try:
        sys_path_insert = BASE / "runtime"
        import sys
        if str(sys_path_insert) not in sys.path:
            sys.path.insert(0,str(sys_path_insert))
        from word_meaning_mapper import explain_text
    except:
        explain_text=lambda x:[]
    for k,v in dedup.items():
        val=v.get("value")
        meanings=[]
        if isinstance(val,str):
            meanings=explain_text(val)
        elif isinstance(val,dict):
            for sub in val.values():
                if isinstance(sub,str):
                    meanings.extend(explain_text(sub))
        enriched[k]={"value":v,"meanings":meanings,"meaning_count":len(meanings)}
    return {"layer":"L2_meaning_enriched","count":len(enriched),"items":enriched}

def build_layer_l3_linked_to_active(enriched):
    # enriched is output from build_layer_l2_meaning_enriched, i.e., dict with 'items'
    enriched_items = enriched.get("items", {}) if isinstance(enriched, dict) else {}
    active_map=_load_json(RUNTIME_LOGS/"semantic_map.json")
    active_index=_load_json(RUNTIME_LOGS/"semantic_index.json")
    active_graph=_load_json(RUNTIME_LOGS/"semantic_graph.json")
    links=[]
    for k,v in enriched_items.items():
        val=v.get("value",{})
        key=val.get("key","")
        # simple linkage by key overlap
        if key in active_map or key in active_index:
            links.append({"archive_key":k,"active_ref":"semantic_map","link_type":"key_match"})
        else:
            # link by token overlap
            tokens=set()
            raw=str(v.get("value",{}).get("value","")).lower().split()
            tokens.update(tokens)
            # check active
            for akey in active_map.keys():
                if any(t in akey.lower() for t in tokens):
                    links.append({"archive_key":k,"active_ref":akey,"link_type":"token_overlap"})
                    break
    return {"layer":"L3_linked_to_active","link_count":len(links),"links":links}

def build_layer_l4_unfold_view(l0,l1,l2,l3):
    # Unfolded view for whole picture
    view={
        "metadata":{
            "built_at":datetime.now(timezone.utc).isoformat(),
            "total_artifacts":l0["count"],
            "dedup_count":l1["count"],
            "enriched_count":l2["count"],
            "links_count":l3["link_count"]
        },
        "layers":{
            "L0_raw_archive":l0,
            "L1_dedup":l1,
            "L2_meaning_enriched":l2,
            "L3_linked_to_active":l3
        },
        "unfold_summary":{
            "archive_is_extension_of_memory":True,
            "active_maps_referenced":["semantic_map.json","semantic_index.json","semantic_graph.json","memory_zones"],
            "recall_policy":"archive is read-only extension; active maps remain primary"
        }
    }
    return view

def build_multilayer_archive_map():
    artifacts=collect_archive_artifacts()
    l0=build_layer_l0_raw(artifacts)
    l1=build_layer_l1_dedup(artifacts)
    l2=build_layer_l2_meaning_enriched(l1["items"])
    l3=build_layer_l3_linked_to_active(l2)
    l4=build_layer_l4_unfold_view(l0,l1,l2,l3)
    _save_json(ARCHIVE_MAP_PATH,l4)
    # Build flat index for fast lookup
    index={}
    for k,v in l2.get("items",{}).items():
        index[k]={"source":v["value"].get("source"),"meaning_count":v["meaning_count"]}
    _save_json(ARCHIVE_INDEX_PATH,index)
    return l4

def unfold_archive_map(query=None):
    map_data=_load_json(ARCHIVE_MAP_PATH)
    if not map_data:
        map_data=build_multilayer_archive_map()
    # simple query expansion
    if query:
        q=query.lower()
        matches=[]
        l2=map_data["layers"]["L2_meaning_enriched"]["items"]
        for k,v in l2.items():
            if q in k.lower() or q in json.dumps(v).lower():
                matches.append(k)
        map_data["query_matches"]=matches
    # link to active maps as extension
    active_maps=[]
    for zone_file in MEMORY_ZONES.glob("*.json"):
        data=_load_json(zone_file)
        if data:
            active_maps.append({"zone":zone_file.name,"count":len(data)})
    map_data["active_zone_summary"]=active_maps
    return map_data

def rehydrate_archive_entries(filter_key=None, max_items=100):
    """Rehydrate selected archived artefacts back into active semantic maps.
    Archive remains source of truth; active maps are extended, not replaced.
    """
    map_data = _load_json(ARCHIVE_MAP_PATH)
    if not map_data:
        map_data = build_multilayer_archive_map()
    l2_items = map_data.get("layers",{}).get("L2_meaning_enriched",{}).get("items",{})
    # pick items
    selected = []
    for k,v in l2_items.items():
        if filter_key and filter_key.lower() not in k.lower():
            continue
        selected.append((k,v))
        if len(selected) >= max_items:
            break
    # Load active maps
    semantic_map_path = RUNTIME_LOGS / "semantic_map.json"
    semantic_index_path = RUNTIME_LOGS / "semantic_index.json"
    active_map = _load_json(semantic_map_path)
    active_index = _load_json(semantic_index_path)
    rehydrated = 0
    for k,v in selected:
        val = v.get("value",{})
        archive_key = val.get("key")
        archive_source = val.get("source")
        if not archive_key:
            continue
        # Skip if still polluted
        if _contains_pollution(val):
            continue
        # Extend active map
        entry = {
            "rehydrated_from_archive": True,
            "archive_key": k,
            "source": archive_source,
            "meanings": v.get("meanings",[]),
            "rehydrated_at": datetime.now(timezone.utc).isoformat()
        }
        # Merge into semantic_map
        if archive_key not in active_map:
            active_map[archive_key] = entry
            rehydrated += 1
        # Also update index if possible
        active_index[archive_key] = {
            "ref": archive_key,
            "rehydrated": True
        }
    _save_json(semantic_map_path, active_map)
    _save_json(semantic_index_path, active_index)
    # Log rehydration
    log_path = RUNTIME_LOGS / "archive_rehydrate_log.json"
    log = _load_json(log_path) or []
    log.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "filter_key": filter_key,
        "requested": len(selected),
        "rehydrated": rehydrated
    })
    _save_json(log_path, log)
    return {"requested": len(selected), "rehydrated": rehydrated}

if __name__=="__main__":
    import sys
    q=" ".join(sys.argv[1:]) if len(sys.argv)>1 else None
    out=unfold_archive_map(q)
    print(json.dumps({"archive_map_built":True,"summary":out["metadata"]},indent=2))
