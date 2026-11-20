"""
Fractal Router with multilayered fractal connections and cross-connects.
Weights are updated by Snapshot Memory success scores.
"""
import json, time, math
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
SNAPSHOT_PATH = BASE_DIR / "infrastructure" / "snapshot_memory" / "snapshots.json"
WEIGHTS_PATH = BASE_DIR / "runtime" / "logs" / "fractal_router_weights.json"

LEVELS = [
    "pre_cache",
    "context_cache",
    "semantic_map",
    "pattern_map",
    "snapshot_memory",
    "hybrid_cognitive_memory",
    "mdwe",
    "weave",
    "reasoning"
]

def _load_weights():
    if WEIGHTS_PATH.exists():
        try:
            return json.loads(WEIGHTS_PATH.read_text())
        except:
            pass
    # default uniform
    w = {}
    for i in range(len(LEVELS)):
        for j in range(len(LEVELS)):
            if i != j:
                w[f"{LEVELS[i]}->{LEVELS[j]}"] = 0.1
    return w

def _save_weights(w):
    WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    WEIGHTS_PATH.write_text(json.dumps(w, indent=2))

_WEIGHTS = _load_weights()

def _load_snapshots():
    if SNAPSHOT_PATH.exists():
        try:
            return json.loads(SNAPSHOT_PATH.read_text())
        except:
            return {}
    return {}

def snapshot_success_for_problem(problem_type):
    snaps = _load_snapshots()
    key = problem_type.lower()
    entry = snaps.get(key)
    if not entry:
        return 0.5
    successes = entry.get("successes",1)
    accuracy = entry.get("accuracy",0.5)
    last_used = entry.get("last_used", time.time())
    age = time.time() - last_used
    recency = 0.5 ** (age / 86400)
    score = min(1.0, successes * accuracy * recency)
    return float(score)

class FractalRouter:
    def __init__(self):
        self.levels = LEVELS
        self.weights = _WEIGHTS

    def route(self, query, diagnostics):
        # Build level scores from diagnostics
        scores = {}
        for lvl in self.levels:
            # heuristic mapping from diagnostics
            if lvl == "pre_cache" and diagnostics.get("pre_cache_hit"):
                scores[lvl] = 1.0
            elif lvl == "context_cache" and diagnostics.get("context_cache_hit"):
                scores[lvl] = 0.9
            elif lvl == "semantic_map" and diagnostics.get("semantic_hit"):
                scores[lvl] = 0.8
            elif lvl == "pattern_map" and diagnostics.get("pattern_hit"):
                scores[lvl] = 0.8
            elif lvl == "snapshot_memory" and diagnostics.get("snapshot_reinforced"):
                scores[lvl] = 0.85
            elif lvl == "hybrid_cognitive_memory":
                scores[lvl] = 0.7
            elif lvl == "mdwe":
                scores[lvl] = 0.6
            elif lvl == "weave":
                scores[lvl] = 0.6
            elif lvl == "reasoning":
                scores[lvl] = 0.5
            else:
                scores[lvl] = 0.1

        # Cross-connect aggregation
        aggregated = {}
        for i, src in enumerate(self.levels):
            agg = scores.get(src,0.0)
            for j, dst in enumerate(self.levels):
                if i == j: continue
                wkey = f"{src}->{dst}"
                w = self.weights.get(wkey,0.1)
                agg += w * scores.get(dst,0.0)
            aggregated[src] = agg

        # Choose best level
        best_level = max(aggregated, key=aggregated.get)
        return {
            "level_scores": scores,
            "aggregated": aggregated,
            "best_level": best_level,
            "best_score": aggregated[best_level]
        }

    def update_weights(self, query, response_source, success=True):
        # Update cross-connect weights using snapshot success
        problem_type = query.split("?")[0].strip()[:80]
        snap_score = snapshot_success_for_problem(problem_type)
        # Reward connections that led to success
        src_level = response_source if response_source in self.levels else "context_cache"
        for dst in self.levels:
            if dst == src_level: continue
            wkey = f"{src_level}->{dst}"
            old = self.weights.get(wkey,0.1)
            # simple RL update
            lr = 0.05 * snap_score
            new = old + lr if success else old - lr * 0.5
            self.weights[wkey] = max(0.01, min(1.0, new))
        _save_weights(self.weights)

def get_fractal_router():
    return FractalRouter()

def route_query_fractal(query, diagnostics):
    router = get_fractal_router()
    res = router.route(query, diagnostics)
    # update weights based on outcome
    if diagnostics.get("response"):
        success = True
        router.update_weights(query, diagnostics.get("response_source","context_cache"), success)
    return res
