"""
Multilayered fixed-size ledger with tree branching for WindowIndexer.

Each layer holds a bounded number of window entries. When a layer fills,
a new layer is created. Sampling is stratified across layers.
"""
from pathlib import Path
import json
from typing import Dict, List

MAX_PER_LAYER = 500_000  # fixed size per layer

class Layer:
    def __init__(self, idx: int, base_dir: Path):
        self.idx = idx
        self.path = base_dir / f"ledger_layer_{idx}.json"
        self.windows: Dict = {}
        self.doc_index: Dict = {}
        self.stats: Dict = {}
        self.curriculum: Dict = {}
        if self.path.exists():
            self.load()

    def load(self):
        with open(self.path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.windows = data.get('windows', {})
        self.doc_index = data.get('doc_index', {})
        self.stats = data.get('stats', {})
        self.curriculum = data.get('curriculum', {})

    def save(self):
        data = {
            'windows': self.windows,
            'doc_index': self.doc_index,
            'stats': self.stats,
            'curriculum': self.curriculum,
        }
        tmp = self.path.with_suffix('.tmp')
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(data, f)
        tmp.replace(self.path)

    def is_full(self):
        return len(self.windows) >= MAX_PER_LAYER

    def record_window(self, window_id: str, meta: dict, doc_id: int, scale: int):
        self.windows[window_id] = meta
        self.doc_index.setdefault(str(doc_id), []).append(window_id)
        self.curriculum.setdefault(str(scale), []).append(window_id)

    def record_sample(self, window_id: str, loss: float, step: int):
        stats = self.stats.get(window_id, {'samples':0, 'sum_loss':0.0, 'last_seen':0})
        stats['samples'] += 1
        stats['sum_loss'] += loss
        stats['last_seen'] = step
        stats['avg_loss'] = stats['sum_loss'] / stats['samples']
        self.stats[window_id] = stats


class MultilayeredWindowLedger:
    def __init__(self, base_path: Path):
        self.base_dir = Path(base_path).parent
        self.layers: List[Layer] = []
        # Load existing layers
        for p in sorted(self.base_dir.glob("ledger_layer_*.json")):
            idx = int(p.stem.split('_')[-1])
            layer = Layer(idx, self.base_dir)
            self.layers.append(layer)
        if not self.layers:
            self.layers.append(Layer(0, self.base_dir))

    def _current_layer(self) -> Layer:
        layer = self.layers[-1]
        if layer.is_full():
            new_idx = len(self.layers)
            layer = Layer(new_idx, self.base_dir)
            self.layers.append(layer)
        return layer

    def record_window(self, window_id: str, meta: dict, doc_id: int, scale: int):
        self._current_layer().record_window(window_id, meta, doc_id, scale)

    def record_sample(self, window_id: str, loss: float, step: int):
        # Find layer containing window
        for layer in self.layers:
            if window_id in layer.stats or window_id in layer.windows:
                layer.record_sample(window_id, loss, step)
                return
        # If not found, record in current layer
        self._current_layer().record_sample(window_id, loss, step)

    def save_all(self):
        for layer in self.layers:
            layer.save()

    def get_curriculum_ids(self, scale: int) -> List[str]:
        scale_key = str(scale)
        ids = []
        for layer in self.layers:
            ids.extend(layer.curriculum.get(scale_key, []))
        return ids

    def get_window_meta(self, window_id: str):
        for layer in self.layers:
            if window_id in layer.windows:
                return layer.windows[window_id]
        return None
