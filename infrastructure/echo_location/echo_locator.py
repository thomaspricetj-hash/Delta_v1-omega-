"""
Echo Location module for Translation OS v1
Pings multiple stores and triangulates location certainty
"""
import json, time, re
from pathlib import Path

BASE = Path("TRANSLATION_OS_V1")
LOG_DIR = BASE / "runtime" / "logs"

def canonicalize(text):
    t = text.lower()
    t = re.sub(r'[^\w\s]', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t

def ping_context_cache(qcanon):
    try:
        import sys
        sys.path.insert(0, str(BASE / "infrastructure" / "context_cache"))
        from context_cache import get_context_cache
        cache = get_context_cache()
        val = cache.get(qcanon)
        return {"store":"context_cache","hit": val is not None, "confidence":0.9 if val else 0.0, "value": val}
    except Exception:
        return {"store":"context_cache","hit":False,"confidence":0.0}

def ping_embeddings(qcanon):
    try:
        import sys
        sys.path.insert(0, str(BASE / "infrastructure" / "embeddings"))
        from embeddings import store as emb_store
        return {"store":"embeddings","hit":True,"confidence":0.7,"value":None}
    except Exception:
        return {"store":"embeddings","hit":False,"confidence":0.0}

def ping_predictor_map(qcanon):
    try:
        map_path = LOG_DIR / "predictor_map.json"
        if map_path.exists():
            pmap = json.loads(map_path.read_text())
            entry = pmap.get(qcanon, {})
            hits = entry.get("hits",0)
            conf = entry.get("last_conf",0.0)
            return {"store":"predictor_map","hit":hits>0,"confidence":conf,"value":entry}
    except Exception:
        pass
    return {"store":"predictor_map","hit":False,"confidence":0.0}

def triangulate(echos):
    scores = [e.get("confidence",0) for e in echos]
    hits = [e.get("hit",False) for e in echos]
    certainty = sum(scores)/len(scores) if scores else 0.0
    agreement = sum(hits)/len(hits) if hits else 0.0
    location_score = 0.7*certainty + 0.3*agreement
    return {"certainty":certainty,"agreement":agreement,"location_score":location_score,"echos":echos}

def echo_locate(query):
    qcanon = canonicalize(query)
    echos = [ping_context_cache(qcanon), ping_embeddings(qcanon), ping_predictor_map(qcanon)]
    result = triangulate(echos)
    log_path = LOG_DIR / "echo_location.jsonl"
    entry = {"ts":time.time(),"query":query,"qcanon":qcanon,**result}
    with open(log_path,"a",encoding="utf-8") as f:
        f.write(json.dumps(entry)+"\n")
    return result

if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "What is the capital of France?"
    print(json.dumps(echo_locate(q), indent=2))
