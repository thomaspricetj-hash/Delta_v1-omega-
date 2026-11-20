"""
Multilayered Foldable Map + Ledger Memory Optimizer
Maps grow → fold to next layer, unfold on view
Ledger drives memory optimization
"""
from collections import defaultdict
from typing import Dict, Any
import json
from pathlib import Path
import sys
BASE = Path(__file__).resolve().parent.parent.parent
STATE_PATH = BASE / "runtime" / "logs" / "foldable_map_state.json"

def _save_state(m):
    try:
        sys.path.insert(0, str(BASE / "scripts"))
        from bitdrop_persistence import save_json
        data = {
            "layers": m.layers,
            "ledger": m.ledger,
            "access_counts": dict(m.access_counts),
            "index": m.index
        }
        save_json(STATE_PATH, data)
    except Exception:
        try:
            with open(STATE_PATH, "w", encoding="utf-8") as f:
                json.dump({"layers": m.layers, "ledger": m.ledger, "access_counts": dict(m.access_counts), "index": m.index}, f)
        except:
            pass

def _load_state():
    if not STATE_PATH.exists():
        return None
    try:
        sys.path.insert(0, str(BASE / "scripts"))
        from bitdrop_persistence import load_json_bitdrop
        data = load_json_bitdrop(STATE_PATH)
        if data:
            return data
    except Exception:
        pass
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return None

class MultilayerFoldableMap:
    def __init__(self, num_layers: int = 3):
        state = _load_state()
        if state:
            self.layers = state.get("layers", [{} for _ in range(num_layers)])
            self.ledger = state.get("ledger", [])
            self.access_counts = defaultdict(int, state.get("access_counts", {}))
            self.index = state.get("index", {})
        else:
            self.layers = [{} for _ in range(num_layers)]
            self.ledger = []
            self.access_counts = defaultdict(int)
            self.index = {}  # key -> layer_id

    def put(self, key: str, value: Any, layer: int = 0):
        self.layers[layer][key] = value
        self.ledger.append({"op":"put","key":key,"layer":layer})
        self.access_counts[key] += 1
        _save_state(self)

    def get(self, key: str):
        for i, layer in enumerate(self.layers):
            if key in layer:
                self.access_counts[key] += 1
                return layer[key], i
        return None, None

    def fold(self, threshold: int = 10):
        # Move low access items to higher layer
        for i in range(len(self.layers)-1):
            to_move = []
            for k,v in list(self.layers[i].items()):
                if self.access_counts[k] < threshold:
                    to_move.append(k)
            for k in to_move:
                v = self.layers[i].pop(k)
                self.layers[i+1][k] = v
                self.ledger.append({"op":"fold","key":k,"from":i,"to":i+1})
        _save_state(self)

    def unfold(self):
        # Return merged view
        merged = {}
        for layer in self.layers:
            merged.update(layer)
        return merged

    def optimize_memory(self):
        # Ledger driven optimization
        self.fold(threshold=5)
        # Compress cold layers
        for i in range(1, len(self.layers)):
            # Placeholder for BitDrop compression
            pass
        _save_state(self)

class LedgerMemoryOptimizer:
    def __init__(self, foldable_map: MultilayerFoldableMap):
        self.map = foldable_map

    def log_access(self, key):
        self.map.access_counts[key] += 1
        self.map.ledger.append({"op":"access","key":key})

    def optimize(self):
        self.map.optimize_memory()

# Singleton
_foldable_map = MultilayerFoldableMap()
_optimizer = LedgerMemoryOptimizer(_foldable_map)

def put(key, value, layer=0):
    _foldable_map.put(key, value, layer)

def get(key):
    return _foldable_map.get(key)

def fold():
    _foldable_map.fold()

def optimize():
    _optimizer.optimize()
