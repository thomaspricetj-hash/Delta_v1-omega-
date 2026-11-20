"""
Fiber Harmony Correlation Engine
Color coding, fibers, micro-fibers, harmony and correlation for routing & visualization.
"""
import re
import json
from pathlib import Path
from collections import defaultdict

BASE = Path(__file__).resolve().parent.parent.parent.parent
STATE_FILE = BASE / "infrastructure" / "fiber_harmony" / "fiber_state.json"

def _load_state():
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except:
            pass
    return {"fibers": {}, "micro_fibers": {}, "harmony_scores": {}, "color_map": {}}

def _save_state(state):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))

COLORS = {
    "high_harmony": "#00FF88",
    "medium_harmony": "#FFD166",
    "low_harmony": "#EF476F",
    "neutral": "#9B9B9B",
    "correlation_strong": "#00BFFF",
    "correlation_weak": "#A0A0A0"
}

def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

def build_fibers(query):
    """Coarse semantic fibers from query -> topic clusters"""
    tokens = tokenize(query)
    state = _load_state()
    fibers = {}
    for t in tokens:
        # simple fiber = first 3 chars + length bucket
        fiber_id = f"f_{t[:3]}_{len(t)}"
        fibers[fiber_id] = fibers.get(fiber_id, 0) + 1
        state["fibers"].setdefault(fiber_id, {"weight":0, "tokens":[]})
        state["fibers"][fiber_id]["weight"] += 1
        if t not in state["fibers"][fiber_id]["tokens"]:
            state["fibers"][fiber_id]["tokens"].append(t)
    _save_state(state)
    return fibers

def build_micro_fibers(query):
    """Fine-grained word-pair micro fibers"""
    tokens = tokenize(query)
    state = _load_state()
    micro = {}
    for i in range(len(tokens)-1):
        pair = f"{tokens[i]}->{tokens[i+1]}"
        micro[pair] = micro.get(pair, 0) + 1
        state["micro_fibers"].setdefault(pair, {"count":0})
        state["micro_fibers"][pair]["count"] += 1
    _save_state(state)
    return micro

def compute_correlation(query_a, query_b):
    """Jaccard correlation over tokens"""
    a = set(tokenize(query_a))
    b = set(tokenize(query_b))
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0

def compute_harmony(query, candidates):
    """
    Harmony = weighted sum of fiber overlap + correlation + micro fiber strength
    candidates: list of dicts with 'answer' field
    """
    state = _load_state()
    q_tokens = set(tokenize(query))
    fibers = build_fibers(query)
    micro = build_micro_fibers(query)
    scores = []
    for cand in candidates:
        ans = cand.get("answer","")
        ans_tokens = set(tokenize(ans))
        # fiber overlap
        overlap = len(q_tokens & ans_tokens)
        # correlation
        corr = compute_correlation(query, ans)
        # micro fiber match
        micro_score = 0
        for pair, cnt in micro.items():
            if pair in state["micro_fibers"]:
                micro_score += state["micro_fibers"][pair]["count"] * 0.1
        harmony = overlap * 2.0 + corr * 10.0 + micro_score
        scores.append({
            "candidate": cand,
            "harmony": harmony,
            "overlap": overlap,
            "correlation": corr,
            "color": _color_for_harmony(harmony)
        })
    # persist harmony
    key = query[:80]
    state["harmony_scores"][key] = scores
    _save_state(state)
    return scores

def _color_for_harmony(h):
    if h >= 15:
        return COLORS["high_harmony"]
    elif h >= 7:
        return COLORS["medium_harmony"]
    elif h > 0:
        return COLORS["low_harmony"]
    return COLORS["neutral"]

def get_color_coding(query):
    state = _load_state()
    key = query[:80]
    scores = state["harmony_scores"].get(key, [])
    if not scores:
        return {"color": COLORS["neutral"], "harmony": 0}
    best = max(scores, key=lambda x: x["harmony"])
    return {"color": best["color"], "harmony": best["harmony"], "candidate": best["candidate"]}

def update_fiber_weights(query, answer, reward=1.0):
    state = _load_state()
    fibers = build_fibers(query)
    for fid, w in fibers.items():
        state["fibers"][fid]["weight"] += reward * w
    # micro fibers
    build_micro_fibers(query)
    build_micro_fibers(answer)
    _save_state(state)

def get_fiber_snapshot():
    state = _load_state()
    return {
        "fiber_count": len(state["fibers"]),
        "micro_fiber_count": len(state["micro_fibers"]),
        "harmony_entries": len(state["harmony_scores"])
    }

def overlay_maps(semantic_score, harmony_score):
    """Combine semantic and harmony into heatmap overlay"""
    combined = 0.7 * semantic_score + 0.3 * harmony_score
    if combined >= 12:
        color = COLORS["high_harmony"]
    elif combined >= 6:
        color = COLORS["medium_harmony"]
    else:
        color = COLORS["low_harmony"]
    return {"combined_score": combined, "color": color}
