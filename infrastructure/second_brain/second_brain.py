"""
Second Brain module for Translation OS v1
Runs a parallel reasoning path and cross-validates with primary brain
"""
import json, time
from pathlib import Path

BASE = Path("TRANSLATION_OS_V1")
LOG_DIR = BASE / "runtime" / "logs"

def canonicalize(text):
    import re
    t = text.lower()
    t = re.sub(r'[^\w\s]', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t

def second_brain_answer(query):
    # Simple second path: use context cache + semantic similarity only
    # This is a lightweight mirror of primary brain
    import sys
    sys.path.insert(0, str(BASE / "infrastructure" / "context_cache"))
    from context_cache import get_context_cache
    cache = get_context_cache()
    qcan = canonicalize(query)
    # Try direct cache
    ans = cache.get(qcan)
    if ans:
        return {"source":"second_brain_cache","response":ans,"confidence":0.9}
    # Try semantic similarity via embeddings
    try:
        sys.path.insert(0, str(BASE / "infrastructure" / "embeddings"))
        from embeddings import store as emb_store
        best_key = None
        best_sim = -1
        for k in list(cache._store.keys())[:200]:
            sim = emb_store.cosine(qcan, k)
            if sim > best_sim:
                best_sim = sim
                best_key = k
        if best_sim > 0.85:
            ans = cache.get(best_key)
            if ans:
                return {"source":"second_brain_semantic","response":ans,"confidence":best_sim}
    except Exception:
        pass
    return {"source":"second_brain_none","response":None,"confidence":0.0}

def cross_validate(query, primary_result):
    secondary = second_brain_answer(query)
    primary_resp = primary_result.get("response")
    secondary_resp = secondary.get("response")
    agree = bool(primary_resp and secondary_resp and str(primary_resp).strip().lower() == str(secondary_resp).strip().lower())
    confidence = (primary_result.get("predictor_confidence",0.5) + secondary.get("confidence",0.0))/2
    verdict = "agree" if agree else ("disagree" if secondary_resp else "secondary_unknown")
    # Log
    log_path = LOG_DIR / "second_brain_crossval.jsonl"
    entry = {
        "ts": time.time(),
        "query": query,
        "primary_source": primary_result.get("response_source"),
        "primary_resp": primary_resp,
        "secondary_source": secondary.get("source"),
        "secondary_resp": secondary_resp,
        "agree": agree,
        "confidence": confidence,
        "verdict": verdict
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry)+"\n")
    return {"agree":agree,"confidence":confidence,"verdict":verdict,"secondary":secondary}

if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "What is the capital of France?"
    print(json.dumps(cross_validate(q, {"response":"Paris","response_source":"test"}), indent=2))
