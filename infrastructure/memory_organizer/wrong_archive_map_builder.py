"""
Wrong Archive Map Builder
Mirror of archive_map_builder for wrong answers / mistakes.
Layers expand, unfold, and overlay with correct archive map for reward steering.
"""
import json
from pathlib import Path
from datetime import datetime, timezone

BASE = Path(__file__).resolve().parent.parent.parent.parent
RUNTIME_LOGS = BASE / "runtime" / "logs"
ARCHIVE_DIR = RUNTIME_LOGS / "memory_archive"
MEMORY_ZONES = RUNTIME_LOGS / "memory_zones"

WRONG_MAP_PATH = RUNTIME_LOGS / "wrong_archive_map_multilayer.json"
WRONG_INDEX_PATH = RUNTIME_LOGS / "wrong_archive_index.json"
OVERLAY_PATH = RUNTIME_LOGS / "archive_overlay_map.json"

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

def collect_wrong_artifacts():
    artifacts = {}
    # Wrong answers are assumed to be polluted artefacts or entries with low harmony/reward
    for f in ARCHIVE_DIR.glob("*.json"):
        data = _load_json(f)
        if not isinstance(data, dict):
            continue
        wrong_items = {}
        for k,v in data.items():
            if _contains_pollution(v):
                wrong_items[k] = v
        if wrong_items:
            artifacts[f.name] = wrong_items
    # Also scan semantic logs for explicit wrong flags
    learning_log = RUNTIME_LOGS / "learning_boost_log.json"
    if learning_log.exists():
        try:
            log = _load_json(learning_log)
            if isinstance(log, list):
                wrong_entries = {}
                for entry in log:
                    if isinstance(entry, dict) and entry.get("correct") is False:
                        q = entry.get("query","")
                        a = entry.get("answer","")
                        if q:
                            wrong_entries[q] = {"answer":a,"reason":entry.get("reason")}
                if wrong_entries:
                    artifacts["learning_boost_wrong.json"] = wrong_entries
        except:
            pass
    return artifacts

def build_layer_l0_raw(artifacts):
    return {"layer":"L0_raw_wrong","count":sum(len(v) for v in artifacts.values()),"artifacts":artifacts}

def build_layer_l1_dedup(artifacts):
    seen=set()
    dedup={}
    for name, data in artifacts.items():
        for k,v in data.items():
            key=f"{name}::{k}"
            if key not in seen:
                seen.add(key)
                dedup[key]={"key":k,"value":v,"source":name}
    return {"layer":"L1_dedup_wrong","count":len(dedup),"items":dedup}

def build_layer_l2_meaning_enriched(dedup):
    enriched={}
    try:
        import sys
        sys.path.insert(0,str(BASE/"runtime"))
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
        enriched[k]={"value":v,"meanings":meanings,"meaning_count":len(meanings),"penalty":1.0}
    return {"layer":"L2_meaning_enriched_wrong","count":len(enriched),"items":enriched}

def build_layer_l3_linked_to_active(enriched):
    enriched_items = enriched.get("items", {}) if isinstance(enriched, dict) else {}
    active_map=_load_json(RUNTIME_LOGS/"semantic_map.json")
    links=[]
    for k,v in enriched_items.items():
        val=v.get("value",{})
        key=val.get("key","")
        if key in active_map:
            links.append({"wrong_key":k,"active_ref":key,"link_type":"collision"})
    return {"layer":"L3_linked_to_active_wrong","link_count":len(links),"links":links}

def build_layer_l4_unfold_view(l0,l1,l2,l3):
    return {
        "metadata":{
            "built_at":datetime.now(timezone.utc).isoformat(),
            "total_wrong_artifacts":l0["count"],
            "dedup_count":l1["count"],
            "enriched_count":l2["count"],
            "links_count":l3["link_count"]
        },
        "layers":{
            "L0_raw_wrong":l0,
            "L1_dedup_wrong":l1,
            "L2_meaning_enriched_wrong":l2,
            "L3_linked_to_active_wrong":l3
        }
    }

def build_wrong_archive_map():
    artifacts=collect_wrong_artifacts()
    l0=build_layer_l0_raw(artifacts)
    l1=build_layer_l1_dedup(artifacts)
    l2=build_layer_l2_meaning_enriched(l1["items"])
    l3=build_layer_l3_linked_to_active(l2)
    l4=build_layer_l4_unfold_view(l0,l1,l2,l3)
    _save_json(WRONG_MAP_PATH,l4)
    index={}
    for k,v in l2.get("items",{}).items():
        index[k]={"source":v["value"].get("source"),"penalty":v.get("penalty",1.0)}
    _save_json(WRONG_INDEX_PATH,index)
    return l4

def overlay_maps(correct_map_path=None, wrong_map_path=None):
    correct_path = Path(correct_map_path) if correct_map_path else RUNTIME_LOGS/"archive_map_multilayer.json"
    wrong_path = Path(wrong_map_path) if wrong_map_path else WRONG_MAP_PATH
    correct = _load_json(correct_path) or {}
    wrong = _load_json(wrong_path) or {}
    # Build overlay with reward/prediction/weights
    overlay = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "correct_summary": correct.get("metadata",{}),
        "wrong_summary": wrong.get("metadata",{}),
        "overlay_layers": {},
        "reward_weights": {},
        "prediction_scores": {}
    }
    # Simple overlay by key overlap
    correct_items = correct.get("layers",{}).get("L2_meaning_enriched",{}).get("items",{})
    wrong_items = wrong.get("layers",{}).get("L2_meaning_enriched_wrong",{}).get("items",{})
    for ck, cv in correct_items.items():
        # Find matching wrong by key similarity
        ckey = cv.get("value",{}).get("key","").lower()
        for wk, wv in wrong_items.items():
            wkey = wv.get("value",{}).get("key","").lower()
            if ckey and wkey and ckey.split()[0] == wkey.split()[0]:
                reward = 1.5
                penalty = wv.get("penalty",1.0)
                weight = reward / (penalty + 0.1)
                overlay["overlay_layers"][ck] = {
                    "correct_key": ck,
                    "wrong_key": wk,
                    "reward": reward,
                    "penalty": penalty,
                    "steer_weight": weight
                }
                overlay["reward_weights"][ck] = weight
                # prediction score: higher weight = more likely to steer right
                overlay["prediction_scores"][ck] = min(1.0, weight/5.0)
                break
    _save_json(OVERLAY_PATH, overlay)
    return overlay

def unfold_wrong_map(query=None):
    data = _load_json(WRONG_MAP_PATH) or build_wrong_archive_map()
    if query:
        q=query.lower()
        matches=[]
        items=data.get("layers",{}).get("L2_meaning_enriched_wrong",{}).get("items",{})
        for k,v in items.items():
            if q in k.lower() or q in json.dumps(v).lower():
                matches.append(k)
        data["query_matches"]=matches
    # overlay with correct map
    overlay=overlay_maps()
    data["overlay_summary"]=overlay.get("correct_summary",{})
    return data

if __name__=="__main__":
    import sys
    build_wrong_archive_map()
    overlay_maps()
    print(json.dumps({"wrong_map_built":True},indent=2))
