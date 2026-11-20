"""
Convergence Map
Tracks where past and future knowledge intersect and emergence happens.
Provides weighted convergence scoring, decay, clustering, and reward propagation.
"""
import json
import time
from pathlib import Path
from collections import defaultdict

BASE = Path(__file__).resolve().parent.parent.parent
STATE_PATH = BASE / "runtime" / "logs" / "convergence_map.json"

def load_map():
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text())
        except:
            pass
    return {"intersections": [], "clusters": {}, "term_weights": {}, "term_index": {}, "query_index": {}, "last_update": 0}

def save_map(state):
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state))

def decay_weights(state, half_life_days=7):
    now = time.time()
    last = state.get("last_update", now)
    if last == now:
        return
    age_days = (now - last) / 86400
    decay = 0.5 ** (age_days / half_life_days)
    for term, w in list(state.get("term_weights", {}).items()):
        state["term_weights"][term] = w * decay
    state["last_update"] = now

def update_convergence(query, past_terms, future_terms, emergence_score, fiber_score=0.5, confidence=0.5):
    state = load_map()
    decay_weights(state, half_life_days=7)
    intersection = set(past_terms) & set(future_terms)
    # weighted score
    score = emergence_score * (0.5 + 0.5 * fiber_score) * (0.5 + 0.5 * confidence)
    entry = {
        "query": query,
        "intersection_terms": list(intersection),
        "emergence_score": emergence_score,
        "weighted_score": score,
        "fiber_score": fiber_score,
        "confidence": confidence,
        "ts": time.time()
    }
    state["intersections"].append(entry)
    # keep last 2000
    state["intersections"] = state["intersections"][-2000:]
    # maintain indexes
    term_index = state.setdefault("term_index", {})
    query_index = state.setdefault("query_index", {})
    qhash = hash(query)
    query_index[qhash] = entry
    for term in intersection:
        term_index.setdefault(term, []).append(qhash)
    # update term weights
    tw = state.setdefault("term_weights", {})
    for term in intersection:
        tw[term] = tw.get(term, 0) + score
    # cluster by first term
    if intersection:
        rep = sorted(intersection)[0]
        cluster = state.setdefault("clusters", {}).setdefault(rep, {"queries": [], "total_score": 0.0, "count": 0})
        cluster["queries"].append(query)
        cluster["total_score"] += score
        cluster["count"] += 1
        # keep last 200 queries per cluster
        cluster["queries"] = cluster["queries"][-200:]
    save_map(state)
    return entry

def get_top_convergences(n=10):
    state = load_map()
    items = sorted(state["intersections"], key=lambda x: x.get("weighted_score",0), reverse=True)
    return items[:n]

def get_hot_terms(n=20):
    state = load_map()
    tw = state.get("term_weights", {})
    return sorted(tw.items(), key=lambda x: x[1], reverse=True)[:n]

def get_cluster_summary():
    state = load_map()
    clusters = state.get("clusters", {})
    summary = []
    for rep, data in clusters.items():
        avg = data["total_score"] / max(1, data["count"])
        summary.append({"representative": rep, "count": data["count"], "avg_score": avg})
    summary.sort(key=lambda x: x["avg_score"], reverse=True)
    return summary[:20]

def get_by_term(term):
    state = load_map()
    term = term.lower()
    index = state.get("term_index", {})
    qhashes = index.get(term, [])
    # return recent intersections for this term
    results = []
    for qhash in qhashes[-50:]:
        entry = state.get("query_index", {}).get(qhash)
        if entry:
            results.append(entry)
    # sort by weighted score
    results.sort(key=lambda x: x.get("weighted_score",0), reverse=True)
    return results

def add_reward_signal(query: str, answer: str, reward: float):
    """Log reward/punishment for query-answer pair and update convergence weights."""
    state = load_map()
    decay_weights(state, half_life_days=7)
    # Simple term extraction
    terms = set(query.lower().split()) | set(answer.lower().split())
    entry = {
        "query": query,
        "answer": answer,
        "reward": reward,
        "terms": list(terms),
        "ts": time.time()
    }
    state.setdefault("reward_log", []).append(entry)
    state["reward_log"] = state["reward_log"][-5000:]
    # Update term weights with reward
    tw = state.setdefault("term_weights", {})
    for term in terms:
        tw[term] = tw.get(term, 0.0) + reward * 0.1
    # Update convergence intersection for mapping
    # Use reward as emergence_score proxy
    update_convergence(query, list(terms), list(terms), emergence_score=abs(reward), fiber_score=0.5, confidence=max(0.1, min(1.0, 0.5 + reward*0.1)))
    save_map(state)
    return entry
