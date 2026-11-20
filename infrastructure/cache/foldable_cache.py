"""
Foldable layered cache for Translation OS
L1 hot memory, L2 BitDrop v2, L3 master map
"""
from pathlib import Path
import sys, json
BASE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE / "scripts"))
from bitdrop_persistence import save_json, load_json_bitdrop

L1_PATH = BASE / "runtime" / "cache" / "l1_cache.json"
L2_PATH = BASE / "runtime" / "logs" / "response_cache_v0.json"

class FoldableCache:
    def __init__(self):
        self.l1 = {}
        self._load_l1()
        self.l2 = load_json_bitdrop(L2_PATH) or {}
    
    def _load_l1(self):
        if L1_PATH.exists():
            try:
                self.l1 = json.loads(L1_PATH.read_text())
            except:
                self.l1 = {}
    
    def _save_l1(self):
        L1_PATH.parent.mkdir(parents=True, exist_ok=True)
        L1_PATH.write_text(json.dumps(self.l1))
    
    def get(self, key):
        # L1 hot
        if key in self.l1:
            return self.l1[key], "l1"
        # L2 warm
        if key in self.l2:
            # promote to L1
            self.l1[key] = self.l2[key]
            self._save_l1()
            return self.l2[key], "l2"
        return None, None
    
    def put(self, key, value, layer="l1"):
        if layer == "l1":
            self.l1[key] = value
            self._save_l1()
        elif layer == "l2":
            self.l2[key] = value
            save_json(L2_PATH, self.l2)
        # fold into master map
        try:
            sys.path.insert(0, str(BASE / "infrastructure"))
            from master_map_integrator import integrate_all
            integrate_all()
        except Exception:
            pass
    
    def fold(self):
        # Merge L1 into L2, keep best
        merged = dict(self.l2)
        merged.update(self.l1)
        save_json(L2_PATH, merged)
        # clear L1
        self.l1 = {}
        self._save_l1()
        return merged

_cache = None
def get_cache():
    global _cache
    if _cache is None:
        _cache = FoldableCache()
    return _cache
