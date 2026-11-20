"""
Dynamic Multi-Layer Weave Lattice Attention
Layers expand as data grows, attention is braided across layers
"""
import json
from pathlib import Path
from collections import defaultdict

BASE = Path('C:/Users/thoma/.lmstudio/apps/bionic/projects/d49037d8-47f4-5808-9028-c707de117f8f/workspace/TRANSLATION_OS_V1')
WEAVE_INDEX = BASE / 'runtime' / 'third_brain' / 'weave_index.json'
LAYER_META = BASE / 'runtime' / 'weave_attention_layers.json'

class WeaveLatticeAttention:
    def __init__(self):
        self.layers = self._load_layers()
        self.index = {}
        if WEAVE_INDEX.exists():
            try:
                self.index = json.loads(WEAVE_INDEX.read_text())
            except Exception:
                self.index = {}

    def _load_layers(self):
        if LAYER_META.exists():
            try:
                return json.loads(LAYER_META.read_text())
            except Exception:
                return {'layers': []}
        return {'layers': []}

    def _save_layers(self):
        LAYER_META.write_text(json.dumps(self.layers))

    def expand_layers(self, query_key, connection_density):
        # Expand layers as attention grows
        layers = self.layers.get('layers', [])
        # Simple heuristic: if density > threshold, add layer
        if not layers:
            layers = [{'id':0, 'capacity':1000, 'keys': []}]
        # Check if need new layer
        need_new = False
        for l in layers:
            if query_key in l.get('keys', []):
                # already present
                need_new = False
                break
        if need_new or connection_density > 0.7:
            new_id = len(layers)
            layers.append({'id': new_id, 'capacity': 1000*(2**new_id), 'keys': [query_key]})
        self.layers['layers'] = layers
        self._save_layers()

    def attend(self, query_key, top_k=5):
        # Braided attention across layers
        scores = []
        for layer in self.layers.get('layers', []):
            # Simple presence score
            if query_key in layer.get('keys', []):
                scores.append((layer['id'], 1.0))
            else:
                # similarity via weave index
                weave = self.index.get('weave', {})
                if query_key in weave:
                    # count connections
                    conn = len(weave[query_key].get('layers', []))
                    score = min(1.0, conn/10.0)
                    scores.append((layer['id'], score))
        # Sort by score desc
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def braid_and_attend(self, query_keys):
        # Multi-key braiding
        combined = defaultdict(float)
        for qk in query_keys:
            for layer_id, score in self.attend(qk):
                combined[layer_id] += score
        # Normalize
        total = sum(combined.values()) or 1.0
        return {lid: score/total for lid, score in combined.items()}

def get_weave_attention():
    return WeaveLatticeAttention()
