"""
Response Synthesizer – merges multi-signal scores into final answer ranking
Combines Rule Book, Convergence, GPS/North Star, Evaluator, Critic, Fiber Harmony
"""
def score_candidate(candidate, diagnostics):
    # candidate is dict with response, source
    score = 0.0
    weights = {
        "rulebook": 0.25,
        "convergence": 0.15,
        "north_star": 0.15,
        "evaluator": 0.20,
        "critic": 0.10,
        "fiber": 0.15
    }
    # Rule Book
    if diagnostics.get("response_source") == "rulebook":
        score += weights["rulebook"] * 1.0
    # Convergence hits
    conv = diagnostics.get("convergence_preload", {})
    hits = conv.get("hits", 0)
    score += weights["convergence"] * min(1.0, hits/5.0)
    # North Star alignment
    ns = diagnostics.get("north_star_alignment", {})
    if isinstance(ns, dict):
        alignment = ns.get("alignment", 0.5)
        score += weights["north_star"] * alignment
    # Evaluator quality
    ev = diagnostics.get("evaluation", {})
    if ev:
        quality = ev.get("quality_score", 0.5)
        score += weights["evaluator"] * quality
    # Critic divergence penalty
    if diagnostics.get("critic_divergent"):
        score -= weights["critic"] * 0.5
    else:
        score += weights["critic"] * 0.5
    # Fiber Harmony
    fiber = diagnostics.get("fiber_harmony_score", 0.5)
    score += weights["fiber"] * fiber
    return round(score, 4)

def synthesize(candidates, diagnostics):
    if not candidates:
        return None
    scored = []
    for c in candidates:
        s = score_candidate(c, diagnostics)
        c2 = dict(c)
        c2["synthesizer_score"] = s
        scored.append(c2)
    best = max(scored, key=lambda c: c["synthesizer_score"])
    return best, scored
