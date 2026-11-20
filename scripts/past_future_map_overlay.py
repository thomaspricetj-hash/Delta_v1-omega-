"""
Past-Future Map Overlay
Computes emergence intersection between past reasoning traces and future plan
"""
import json
from pathlib import Path
BASE = Path(__file__).resolve().parent.parent

TRACE_PATH = BASE / "runtime" / "logs" / "reasoning_traces.jsonl"
PLAN_PATH = BASE / "runtime" / "logs" / "project_plan.json"
CLUSTER_PATH = BASE / "runtime" / "logs" / "query_clusters.json"

from bitdrop_persistence import load_json_bitdrop, load_lines_bitdrop

def _load_recent_traces(n=200):
    # Try BitDrop first
    lines = load_lines_bitdrop(TRACE_PATH)
    if not lines:
        if TRACE_PATH.exists():
            try:
                lines = []
                for ln in TRACE_PATH.read_text(encoding='utf-8', errors='ignore').splitlines():
                    if not ln.strip(): continue
                    try: lines.append(json.loads(ln).get('data', json.loads(ln)))
                    except: pass
            except: pass
    recent = lines[-n:] if lines else []
    return recent

def _load_future_plan():
    plan = load_json_bitdrop(PLAN_PATH)
    if not plan and PLAN_PATH.exists():
        try: plan = json.loads(PLAN_PATH.read_text(encoding='utf-8'))
        except: plan = {}
    return plan or {}

def _load_clusters():
    data = load_json_bitdrop(CLUSTER_PATH)
    if not data and CLUSTER_PATH.exists():
        try: data = json.loads(CLUSTER_PATH.read_text(encoding='utf-8'))
        except: data = {}
    return data or {}

def overlay_past_future(query, diagnostics):
    past_traces = _load_recent_traces(200)
    future_plan = _load_future_plan()
    clusters = _load_clusters()

    # Build past term frequencies with rewards
    from collections import Counter
    past_counter = Counter()
    for t in past_traces:
        q = (t.get('query') or '').lower()
        past_counter.update(q.split())
    # Add reward from reasoning traces if present
    for t in past_traces:
        # reward boost for high fiber/confidence traces
        fiber = t.get('fiber_harmony_score') or 0.5
        conf = t.get('confidence_prior') or 0.5
        reward = 0.5*fiber + 0.5*conf
        q = (t.get('query') or '').lower()
        for w in q.split():
            past_counter[w] += reward

    past_terms = set(past_counter.keys())

    # Current micro-cluster terms
    cur_cluster_terms = set()
    cg = diagnostics.get('chunking_info') or {}
    priority = diagnostics.get('day_dream_priority') or {}
    rep = priority.get('representative') or ''
    cur_cluster_terms.update(rep.lower().split())
    # Present weight from fiber harmony
    fiber_present = diagnostics.get('fiber_harmony_score') or 0.5
    conf_present = diagnostics.get('confidence_prior') or 0.5
    present_weight = 0.6*fiber_present + 0.4*conf_present

    # Future terms with priority weights
    future_terms_set = set()
    future_weights = Counter()
    next_actions = future_plan.get('next_actions', []) or []
    # Higher weight for earlier actions
    for idx, act in enumerate(next_actions[:10]):
        w = 1.0 / (idx+1)
        words = str(act).lower().split()
        future_terms_set.update(words)
        for word in words:
            future_weights[word] += w
    milestones = future_plan.get('milestones', []) or []
    for m in milestones:
        name = str(m.get('name','')).lower()
        for w in name.split():
            future_terms_set.add(w)
            future_weights[w] += 0.5

    future_terms = future_terms_set

    # Intersection with weights
    intersection = past_terms & cur_cluster_terms & future_terms
    weighted_score = 0.0
    term_rewards = {}
    for term in intersection:
        past_w = past_counter.get(term, 1)
        future_w = future_weights.get(term, 0.1) + 0.1
        # Reward push: boost terms that are in intersection
        reward = (0.4 * past_w + 0.4 * future_w + 0.2 * present_weight)
        # Apply non-linear push
        reward = reward ** 1.2
        weighted_score += reward
        term_rewards[term] = reward

    # Emergence score with weighted rewards
    emergence_score = weighted_score
    if intersection:
        fiber = diagnostics.get('fiber_harmony_score') or 0.5
        conf = diagnostics.get('confidence_prior') or 0.5
        emergence_score *= (0.5*fiber + 0.5*conf)

    # Past map summary
    past_map = {
        'trace_count': len(past_traces),
        'unique_terms': len(past_terms),
        'recent_queries': [t.get('query','')[:120] for t in past_traces[-5:]]
    }
    # Future map summary
    future_map = {
        'next_actions': next_actions[:5],
        'milestone_count': len(milestones)
    }
    # Present map summary
    present_map = {
        'micro_clusters': cg.get('num_micro_clusters'),
        'cluster_id': cg.get('micro_cluster_id'),
        'priority': priority
    }

    overlay = {
        'past_map': past_map,
        'present_map': present_map,
        'future_map': future_map,
        'intersection_terms': list(intersection),
        'term_rewards': term_rewards,
        'weighted_score': weighted_score,
        'emergence_score': emergence_score,
        'query': query,
        'weights': {
            'present_weight': present_weight,
            'past_term_count': len(past_counter),
            'future_term_count': len(future_terms)
        }
    }
    return overlay

if __name__ == '__main__':
    print(json.dumps(overlay_past_future('test', {}), indent=2))
