"""
Domain Map Builder
Builds multilayer maps for math, physics, science, biology, conversation
Same architecture as archive map: L0 raw, L1 dedup, L2 meaning enriched, L3 linked, L4 unfold
"""
import json
from pathlib import Path
from datetime import datetime, timezone

BASE = Path(__file__).resolve().parent.parent.parent
RUNTIME_LOGS = BASE / "runtime" / "logs"

def _load_json(p):
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    except:
        return {}

def _save_json(p, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")

def build_domain_map(domain_name, source_data):
    """
    domain_name: str e.g. 'math', 'physics'
    source_data: dict of {key: value}
    """
    # L0 raw
    l0 = {"layer": f"L0_raw_{domain_name}", "count": len(source_data), "items": source_data}
    # L1 dedup
    seen=set()
    dedup={}
    for k,v in source_data.items():
        key=f"{k}::{json.dumps(v,sort_keys=True)[:200]}"
        if key not in seen:
            seen.add(key)
            dedup[k]={"key":k,"value":v}
    l1 = {"layer": f"L1_dedup_{domain_name}", "count": len(dedup), "items": dedup}
    # L2 meaning enriched
    try:
        import sys
        sys.path.insert(0,str(BASE/"runtime"))
        from word_meaning_mapper import explain_text
    except:
        explain_text=lambda x:[]
    enriched={}
    for k,v in dedup.items():
        meanings=[]
        raw=str(v.get("value",""))
        meanings=explain_text(raw)
        enriched[k]={"value":v,"meanings":meanings,"meaning_count":len(meanings)}
    l2 = {"layer": f"L2_meaning_enriched_{domain_name}", "count": len(enriched), "items": enriched}
    # L3 linked to active
    active_map=_load_json(RUNTIME_LOGS/"semantic_map.json")
    links=[]
    for k,v in enriched.items():
        if k in active_map:
            links.append({"domain_key":k,"active_ref":k,"link_type":"direct"})
    l3 = {"layer": f"L3_linked_{domain_name}", "link_count": len(links), "links": links}
    # L4 unfold
    view={
        "domain": domain_name,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "metadata": {
            "raw_count": l0["count"],
            "dedup_count": l1["count"],
            "enriched_count": l2["count"],
            "links_count": l3["link_count"]
        },
        "layers": {
            "L0_raw": l0,
            "L1_dedup": l1,
            "L2_meaning_enriched": l2,
            "L3_linked": l3
        }
    }
    out_path = RUNTIME_LOGS / f"{domain_name}_domain_map_multilayer.json"
    _save_json(out_path, view)
    return view

def build_all_domain_maps():
    # Collect source data from existing maps / training logs
    semantic_map = _load_json(RUNTIME_LOGS/"semantic_map.json") or {}
    # Heuristic categorization by query keywords
    domains = {
        "math": {},
        "physics": {},
        "science": {},
        "biology": {},
        "conversation": {}
    }
    for k,v in semantic_map.items():
        kl = k.lower()
        if any(w in kl for w in ["derivative","integral","x^","^","pi","square","prime"]):
            domains["math"][k]=v
        elif any(w in kl for w in ["force","energy","velocity","acceleration","newton","joule","watt"]):
            domains["physics"][k]=v
        elif any(w in kl for w in ["chemical","element","atomic","boiling","melting","formula"]):
            domains["science"][k]=v
        elif any(w in kl for w in ["cell","dna","gene","organism","species","evolution"]):
            domains["biology"][k]=v
        else:
            domains["conversation"][k]=v
    results={}
    for dom, data in domains.items():
        if data:
            results[dom]=build_domain_map(dom, data)
        else:
            # create empty map
            results[dom]=build_domain_map(dom, {})
    return results

if __name__=="__main__":
    res=build_all_domain_maps()
    print(json.dumps({k:v["metadata"] for k,v in res.items()},indent=2))
