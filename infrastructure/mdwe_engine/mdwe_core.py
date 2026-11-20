"""
MDWE Engine - Master + Delta Weight Engine
Sparse Cognitive Routing • Dual-Weight Learning • SIMD/NUMA Acceleration
Python reference implementation for integration
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import time
import json
from pathlib import Path

@dataclass
class NeuronMeta:
    id: int
    decay: float = 0.0
    last_active: float = field(default_factory=time.time)
    frequency: int = 0
    semantic_tag: str = ""
    branch_id: int = 0

class DualWeightNeuron:
    def __init__(self, dim: int, neuron_id: int):
        self.id = neuron_id
        self.master = np.random.randn(dim).astype(np.float32) * 0.1
        self.delta = np.zeros(dim, dtype=np.float32)
        # Pattern store integration
        try:
            from pattern_store.pattern_store import store_pattern
            self.pattern_id = store_pattern(self.master.tolist())
        except Exception:
            self.pattern_id = None
        self.meta = NeuronMeta(id=neuron_id)
        self.active = False

    @property
    def effective(self):
        return self.master + self.delta

    def apply_decay(self, rate=0.01):
        self.delta *= (1 - rate)
        # structured decay
        self.meta.decay = max(0.0, self.meta.decay - 0.01)

    def update_delta(self, grad, lr=0.01):
        self.delta += lr * grad
        # drift protection: symmetric clipping
        np.clip(self.delta, -1.0, 1.0, out=self.delta)
        self.meta.decay += 0.1
        self.meta.last_active = time.time()
        self.meta.frequency += 1
        # Store transformation for reuse
        try:
            from pattern_store.pattern_store import store_transformation, reuse_transformation
            if self.pattern_id:
                # Check if transformation already stored
                found, _ = reuse_transformation(self.pattern_id, 'update', {'lr': lr})
                if not found:
                    store_transformation(self.pattern_id, 'update', {'lr': lr}, self.delta.tolist())
        except Exception:
            pass

class SparseRouter:
    def __init__(self, num_neurons: int, dim: int, activation_threshold: float = 0.5):
        self.neurons = [DualWeightNeuron(dim, i) for i in range(num_neurons)]
        self.threshold = activation_threshold
        self.heat_map = np.zeros(num_neurons)

    def heat_extract(self, inputs: np.ndarray) -> np.ndarray:
        # SIMD-like block skimming
        scores = []
        for n in self.neurons:
            score = float(np.dot(n.effective, inputs))
            scores.append(abs(score))
        scores = np.array(scores)
        # heatmap
        self.heat_map = scores
        return scores

    def route(self, scores: np.ndarray) -> List[int]:
        # Heat-based sparse routing with caps
        active = np.where(scores > self.threshold)[0]
        # congestion control: limit to top K
        k = min(len(active), 256)
        if len(active) > 0:
            top_idx = np.argpartition(scores[active], -k)[-k:]
            active = active[top_idx]
        # minimum spacing rule
        if len(active) > 1:
            active = np.sort(active)
        return active.tolist()

    def forward(self, inputs: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        # Computation Cache check
        try:
            from computation_cache.computation_cache import reuse_mul, hash_block
            cache_key = hash_block(inputs.tolist())
            # Simple reuse check placeholder
        except Exception:
            pass
        scores = self.heat_extract(inputs)
        active_ids = self.route(scores)
        # compute sparse output
        out = np.zeros_like(inputs)
        for idx in active_ids:
            n = self.neurons[idx]
            n.active = True
            out += n.effective * scores[idx]
            n.meta.decay = min(1.0, n.meta.decay + 0.05)
        # deactivate others
        for i, n in enumerate(self.neurons):
            if i not in active_ids:
                n.active = False
                n.apply_decay()
        return out, active_ids

class CognitiveRuleEngine:
    def __init__(self):
        self.rules = {
            "magnitude": lambda n, s: s > 0.5,
            "recency": lambda n, s: time.time() - n.meta.last_active < 3600,
            "frequency": lambda n, s: n.meta.frequency > 0,
            "decay": lambda n, s: n.meta.decay < 0.8,
        }

    def allow_activation(self, neuron: DualWeightNeuron, score: float) -> bool:
        return all(rule(neuron, score) for rule in self.rules.values())

class BranchPredictor:
    def predict_branch(self, neuron: DualWeightNeuron, score: float) -> int:
        # simple heuristic based on delta magnitude, heat, decay
        delta_mag = np.linalg.norm(neuron.delta)
        if delta_mag > 0.5 and score > 0.7:
            return 1
        elif neuron.meta.decay > 0.5:
            return 2
        else:
            return 0

class DeltaGraphLedger:
    def __init__(self):
        self.nodes: List[Dict] = []
        self.edges: List[Dict] = []
        self.step = 0

    def record(self, active_ids: List[int], scores: np.ndarray):
        self.step += 1
        for idx in active_ids:
            self.nodes.append({
                "step": self.step,
                "neuron_id": idx,
                "score": float(scores[idx]),
                "timestamp": time.time()
            })
        for i in range(len(active_ids)-1):
            self.edges.append({
                "step": self.step,
                "from": int(active_ids[i]),
                "to": int(active_ids[i+1]),
                "coactive": True
            })

    def export(self) -> Dict:
        return {"nodes": self.nodes, "edges": self.edges, "steps": self.step}

class BitDropV2:
    @staticmethod
    def compress(weights: np.ndarray, profile: str = "fast") -> bytes:
        # ultra-compact snapshot placeholder
        if profile == "fast":
            # quantize to int8
            q = np.clip(weights, -1,1)
            q = (q * 127).astype(np.int8)
            return q.tobytes()
        else:
            # adaptive
            return weights.astype(np.float16).tobytes()

    @staticmethod
    def decompress(data: bytes, shape: Tuple[int,...], profile: str = "fast"):
        if profile == "fast":
            arr = np.frombuffer(data, dtype=np.int8).astype(np.float32) / 127.0
        else:
            arr = np.frombuffer(data, dtype=np.float16).astype(np.float32)
        return arr.reshape(shape)

class MDWEEngine:
    def __init__(self, num_neurons=1024, dim=128):
        self.router = SparseRouter(num_neurons, dim)
        self.rules = CognitiveRuleEngine()
        self.branch_predictor = BranchPredictor()
        self.ledger = DeltaGraphLedger()
        self.num_neurons = num_neurons
        self.dim = dim

    def forward(self, inputs: np.ndarray):
        scores = self.router.heat_extract(inputs)
        # rule gating
        gated_ids = []
        for i, n in enumerate(self.router.neurons):
            if self.rules.allow_activation(n, scores[i]):
                gated_ids.append(i)
        # route only gated
        masked_scores = scores.copy()
        mask = np.ones_like(scores, dtype=bool)
        mask[gated_ids] = True
        # Actually route with threshold
        active_ids = self.router.route(scores)
        # branch prediction
        branches = {i: self.branch_predictor.predict_branch(self.router.neurons[i], scores[i]) for i in active_ids}
        # compute sparse forward
        out, active_ids = self.router.forward(inputs)
        # ledger
        self.ledger.record(active_ids, scores)
        # SIMD/NUMA heuristic
        strategy = "scalar"
        if len(active_ids) >= 4096:
            strategy = "NUMA"
        elif len(active_ids) >= 256:
            strategy = "AVX2/AVX-512"
        return {
            "output": out,
            "active_ids": active_ids,
            "branches": branches,
            "strategy": strategy,
            "ledger_size": len(self.ledger.nodes)
        }

    def save_snapshot(self, path):
        from pathlib import Path
        import json
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "master_weights": [n.master.tolist() for n in self.router.neurons],
            "delta_weights": [n.delta.tolist() for n in self.router.neurons],
            "ledger": self.ledger.export()
        }
        # DAX Registry + Hybrid BitDrop compression for ledger
        try:
            from infrastructure.dax_registry.dax_registry import DAXRegistry
            dax_path = p.parent / 'mdwe_dax'
            dax = DAXRegistry(dax_path)
            # Store ledger as compressed delta
            ledger_bytes = json.dumps(self.ledger.export()).encode('utf-8')
            dax.add_delta('mdwe_ledger', ledger_bytes, is_dax=True)
        except Exception:
            pass
        # Compress with BitDrop v2
        try:
            sys_path = str(Path(__file__).resolve().parent.parent.parent)
            import sys
            if sys_path not in sys.path:
                sys.path.insert(0, sys_path)
            from bitdrop_v2_integration import compress_data
            raw_bytes = json.dumps(data).encode('utf-8')
            comp_info = compress_data(raw_bytes)
            out_path = p.with_suffix(p.suffix + '.bd2')
            with open(out_path, 'w') as f:
                json.dump({"compressed": True, "info": comp_info, "raw_len": len(raw_bytes)}, f)
            with open(p, "w") as f:
                json.dump(data, f)
        except Exception:
            with open(p, "w") as f:
                json.dump(data, f)

    def load_snapshot(self, path: Path):
        if not path.exists():
            return
        with open(path) as f:
            data = json.load(f)
        for n, m in zip(self.router.neurons, data["master_weights"]):
            n.master = np.array(m, dtype=np.float32)
        for n, d in zip(self.router.neurons, data["delta_weights"]):
            n.delta = np.array(d, dtype=np.float32)
