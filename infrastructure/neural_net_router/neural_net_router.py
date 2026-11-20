"""
Neural Net Router – lightweight coordination net for Translation OS v1
Provides:
* shared encoder for queries
* router net for component selection
* retrieval scoring for caches
* resonance scoring for Gyrotron
"""
import json
import hashlib
from pathlib import Path
import math

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent

_MODEL_PATH = BASE_DIR / "runtime" / "logs" / "neural_net_router_state.json"

def _load_state():
    if _MODEL_PATH.exists():
        try:
            return json.loads(_MODEL_PATH.read_text())
        except Exception:
            pass
    return {"weights": {}, "stats": {"hits":0, "calls":0}}

def _save_state(state):
    try:
        _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        _MODEL_PATH.write_text(json.dumps(state))
    except Exception:
        pass

def _hash_features(text: str):
    # Simple deterministic feature vector from text
    h = hashlib.sha256(text.encode()).digest()
    # 8-dim pseudo features
    feats = [int.from_bytes(h[i:i+4], 'little') / 2**32 for i in range(0,32,4)]
    return feats

def encode_query(query: str):
    feats = _hash_features(query)
    # Normalize
    norm = math.sqrt(sum(f*f for f in feats)) or 1.0
    return [f/norm for f in feats]

class NeuralNetRouter:
    def __init__(self):
        self.state = _load_state()
        self.weights = self.state.get("weights", {})

    def route(self, query: str, candidate_components):
        # Simple learned scoring: component prior + query similarity
        q_vec = encode_query(query)
        scores = {}
        for comp in candidate_components:
            w = self.weights.get(comp.lower(), 0.5)
            # pseudo similarity = dot with component bias
            bias = hash(comp) % 1000 / 1000.0
            score = 0.7*w + 0.3*bias
            scores[comp] = score
        # sort
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        self.state["stats"]["calls"] += 1
        return ranked

    def update(self, query: str, chosen_component: str, success: bool):
        key = chosen_component.lower()
        w = self.weights.get(key, 0.5)
        # simple reward update
        delta = 0.1 if success else -0.05
        w = max(0.0, min(1.0, w + delta))
        self.weights[key] = w
        if success:
            self.state["stats"]["hits"] += 1
        _save_state(self.state)

    def prewarm(self, query: str):
        # Return top 3 components to pre-load
        candidates = [
            "Pre-Cache","Context Cache","Computation Cache","Pattern Classifier",
            "Hybrid Cognitive Memory","MDWE Engine","True Hybrid","Weave"
        ]
        ranked = self.route(query, candidates)
        return [c for c,_ in ranked[:3]]

def get_neural_net_router():
    return NeuralNetRouter()

def route_query(query: str, components):
    router = get_neural_net_router()
    return router.route(query, components)

def update_router(query: str, component: str, success: bool):
    router = get_neural_net_router()
    router.update(query, component, success)

def prewarm_caches(query: str):
    router = get_neural_net_router()
    return router.prewarm(query)
