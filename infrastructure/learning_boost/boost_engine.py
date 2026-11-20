"""
Learning Boost Engine
Pushes learning further via reward shaping, uncertainty sampling, paraphrase augmentation, and cross-map consensus.
"""
import json
from pathlib import Path
import random
import re

BASE = Path(__file__).resolve().parent.parent.parent.parent
RUNTIME_LOGS = BASE / "runtime" / "logs"

def paraphrase_query(q):
    """Simple paraphrase augmentation for training diversity"""
    paraphrases = []
    # basic transforms
    if "what is" in q.lower():
        paraphrases.append(q.replace("what is", "what's"))
    if "how many" in q.lower():
        paraphrases.append(q.replace("how many", "how much"))
    # add synonyms
    synonyms = {"capital": ["capital city", "main city"], "derivative": ["d/dx", "differentiate"]}
    for k, v in synonyms.items():
        if k in q.lower():
            for syn in v:
                paraphrases.append(q.replace(k, syn))
    return list(set(paraphrases + [q]))

def reward_decay(weight, age_days=0, half_life=30):
    """Exponential decay for stale memories"""
    import math
    return weight * (0.5 ** (age_days / half_life))

def uncertainty_sample(candidates):
    """Pick candidates with mid-range confidence for active learning"""
    # candidates: list of dict with score
    if not candidates:
        return []
    # sort by score
    scores = [c.get("score",0) for c in candidates]
    # pick those in 40-70 percentile
    sorted_c = sorted(candidates, key=lambda x: x.get("score",0))
    n = len(sorted_c)
    start = int(n * 0.3)
    end = int(n * 0.7)
    return sorted_c[start:end]

def cross_map_consensus(query, maps):
    """Simple voting across maps"""
    votes = {}
    for map_name, data in maps.items():
        if isinstance(data, dict):
            for k, v in data.items():
                if query.lower() in k.lower():
                    ans = str(v)[:200]
                    votes[ans] = votes.get(ans,0)+1
    if not votes:
        return None
    # return most voted
    best = max(votes.items(), key=lambda x: x[1])
    return {"answer": best[0], "votes": best[1], "confidence": best[1]/len(maps)}

def boost_learning(query, answer, source):
    """Log boosted learning event"""
    log_path = RUNTIME_LOGS / "learning_boost_log.json"
    entry = {
        "query": query,
        "answer": answer,
        "source": source,
        "paraphrases": paraphrase_query(query),
        "timestamp": __import__("datetime").datetime.utcnow().isoformat() + "Z"
    }
    data = []
    if log_path.exists():
        try:
            data = json.loads(log_path.read_text())
        except:
            data = []
    data.append(entry)
    # keep last 1000
    data = data[-1000:]
    log_path.write_text(json.dumps(data, indent=2))
    return entry

def apply_boost_to_orchestrator(query, candidates):
    """Return boosted candidates with reward shaping"""
    # uncertainty sampling
    uncertain = uncertainty_sample(candidates)
    # cross map consensus if possible
    # reward shaping: boost candidates with higher confidence
    boosted = []
    for c in candidates:
        score = c.get("score",0)
        # reward shaping
        score *= 1.2 if c.get("source") in ["math_reasoning","fact_kb","fiber_harmony"] else 1.0
        c_copy = dict(c)
        c_copy["score"] = score
        boosted.append(c_copy)
    # sort
    boosted.sort(key=lambda x: x.get("score",0), reverse=True)
    return boosted, uncertain
