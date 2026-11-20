"""
Route Optimizer
Scores routes based on correct answers, weights, rewards, map, ledger, index
"""
from typing import List, Dict

def score_route(route: Dict, diagnostics: Dict, semantic_score: float, reward: float, ledger_ok: bool, index_hit: bool, pattern_hit: bool=False, harmonic_coherence: float=0.0) -> float:
    base = route.get("score", 0.5)
    sem = 0.35 * min(semantic_score/10.0, 1.0)
    rew = 0.20 * min(reward/10.0, 1.0)
    led = 0.10 if ledger_ok else -0.1
    idx = 0.10 if index_hit else 0.0
    pat = 0.05 if pattern_hit else 0.0
    harm = 0.20 * min(max(harmonic_coherence, 0.0), 1.0)
    return base + sem + rew + led + idx + pat + harm

def pick_best_route(candidates: List[Dict], diagnostics: Dict) -> Dict:
    if not candidates:
        return {"route": "model", "score": 0.0}
    best = None
    best_score = -1e9
    semantic_score = diagnostics.get("semantic_score", 0.0)
    reward = diagnostics.get("reward", 0.0)
    ledger_ok = diagnostics.get("ledger_ok", False)
    index_hit = diagnostics.get("index_hit", False)
    pattern_hit = diagnostics.get("pattern_reuse_hit", False)
    harmonic_coherence = diagnostics.get("fiber_harmony_score", 0.0)
    # normalize fiber harmony score to 0-1
    try:
        harmonic_coherence = float(harmonic_coherence)
    except Exception:
        harmonic_coherence = 0.0
    for r in candidates:
        s = score_route(r, diagnostics, semantic_score, reward, ledger_ok, index_hit, pattern_hit, harmonic_coherence)
        if s > best_score:
            best_score = s
            best = r
    if best is None:
        best = candidates[0]
    best = dict(best)
    best["optimizer_score"] = best_score
    return best
