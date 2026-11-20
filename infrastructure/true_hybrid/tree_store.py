"""
Tree-structured codebook / index / ledger with automatic node splitting.
Each node holds up to max_entries items; when full it splits into children.
This gives O(log N) lookup and keeps each node small for fast I/O.
"""

from pathlib import Path
import numpy as np
import json
from typing import List, Optional

class TreeNode:
    def __init__(self, capacity: int, dir_path: Path, name: str):
        self.capacity = capacity
        self.dir = dir_path / name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.entries: List = []
        self.children: List['TreeNode'] = []
        self.file = self.dir / "data.npy"

    def save(self):
        np.save(self.file, np.array(self.entries, dtype=object))

    def load(self):
        if self.file.exists():
            self.entries = list(np.load(self.file, allow_pickle=True))

class TreeStore:
    def __init__(self, capacity_per_node: int, dir_path: str, name: str):
        self.capacity = capacity_per_node
        self.dir = Path(dir_path)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.root = TreeNode(capacity_per_node, self.dir, "node_0000")
        self.root.load()

    def _walk(self):
        stack = [self.root]
        while stack:
            n = stack.pop()
            yield n
            stack.extend(n.children)

    def _find_leaf(self):
        for node in self._walk():
            if len(node.entries) < self.capacity:
                return node
        # create new child under root
        new = TreeNode(self.capacity, self.dir, f"node_{len(list(self._walk())):04d}")
        self.root.children.append(new)
        return new

    def _split_node(self, node: TreeNode):
        # Variance-aware split: if values are similar, keep together; else split
        if len(node.entries) == 0:
            return
        import numpy as np
        vals = np.array(node.entries, dtype=np.float32)
        # If variance is low, just split by count
        # If variance is high, split by median to separate ranges
        if np.var(vals) > 1e-6:
            # split by median for better locality
            mid_idx = int(np.argsort(vals)[len(vals)//2])
            sorted_idx = np.argsort(vals)
            left_vals = [node.entries[i] for i in sorted_idx[:len(sorted_idx)//2]]
            right_vals = [node.entries[i] for i in sorted_idx[len(sorted_idx)//2:]]
        else:
            mid = len(node.entries)//2
            left_vals = node.entries[:mid]
            right_vals = node.entries[mid:]
        node.entries = []
        node.save()
        left = TreeNode(self.capacity, self.dir, f"{node.dir.name}L")
        right = TreeNode(self.capacity, self.dir, f"{node.dir.name}R")
        left.entries = left_vals
        right.entries = right_vals
        left.save()
        right.save()
        node.children = [left, right]

class TreeCodebook(TreeStore):
    def __init__(self, capacity_per_node: int = 1024, dir_path: str = "codebook_tree"):
        super().__init__(capacity_per_node, dir_path, "codebook")
        self.value_to_loc = {}
        self.value_to_code = {}
        self.code_to_value = []
        self.access_counts = {}
        self._load_mapping()

    def _save_mapping(self):
        map_path = self.dir / "code_map.npy"
        np.save(map_path, np.array(self.code_to_value, dtype=np.float32))
        # Persist access counts for frequency-aware splits
        import json
        cnt_path = self.dir / "access_counts.json"
        # Convert float keys to strings for JSON
        cnt_serial = {str(k): int(v) for k,v in self.access_counts.items()}
        with open(cnt_path, 'w') as f:
            json.dump(cnt_serial, f)
    def _load_mapping(self):
        map_path = self.dir / "code_map.npy"
        if map_path.exists():
            arr = np.load(map_path)
            self.code_to_value = list(arr)
            self.value_to_code = {v:i for i,v in enumerate(self.code_to_value)}
        cnt_path = self.dir / "access_counts.json"
        if cnt_path.exists():
            import json
            with open(cnt_path) as f:
                cnt_serial = json.load(f)
            # keys stored as strings
            self.access_counts = {float(k): int(v) for k,v in cnt_serial.items()}
    def save_meta(self, path):
        import json
        meta = {"capacity": self.capacity, "type": "TreeCodebook"}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(meta, f)
        # Persist code mapping for reconstruction
        self._save_mapping()

    def fit_from_data(self, data):
        import numpy as np
        uniq = np.unique(np.asarray(data, dtype=np.float32).flatten())
        for v in uniq:
            self.add(v)
        self._save_mapping()

    def add(self, v):
        v_f = float(v)
        if v_f in self.value_to_code:
            return
        # assign code
        code = len(self.code_to_value)
        self.value_to_code[v_f] = code
        self.code_to_value.append(v_f)
        # store in tree node for persistence
        node = self._find_leaf()
        if len(node.entries) >= self.capacity:
            self._split_node(node)
            node = self._find_leaf()
        node.entries.append(v_f)
        node.save()
        # also keep old mapping for compatibility
        if v_f not in self.value_to_loc:
            idx = len(node.entries)-1
            self.value_to_loc[v_f] = (id(node), idx)

    def _split_node(self, node: 'TreeNode'):
        # Variance-aware split extended with lookup frequency weighting and node size
        if len(node.entries) == 0:
            return
        import numpy as np
        vals = np.array(node.entries, dtype=np.float32)
        # Frequency per entry
        freqs = np.array([float(self.access_counts.get(float(v), 0)) for v in node.entries], dtype=np.float32)
        # Avoid zero freq -> treat as 1
        freqs = np.where(freqs == 0, 1.0, freqs)
        var = float(np.var(vals))
        # Size factor
        size_factor = len(node.entries) / max(1, self.capacity)
        # If variance is low and size small, keep together
        if var <= 1e-6 and size_factor < 0.75:
            # split by count only
            mid = len(node.entries) // 2
            left_vals = node.entries[:mid]
            right_vals = node.entries[mid:]
        else:
            # Combine value similarity and frequency for split point
            # Sort by value
            order = np.argsort(vals)
            vals_sorted = vals[order]
            freqs_sorted = freqs[order]
            # Weighted split by cumulative frequency to keep hot values together
            total_freq = freqs_sorted.sum() + 1e-9
            # Variance-based bias: higher variance -> prefer value-based split
            variance_bias = min(1.0, var / 1e-3)  # scale
            # Blend value order and frequency: use frequency-weighted median
            cum_freq = np.cumsum(freqs_sorted)
            # Target split at half total frequency, adjusted by variance
            target = total_freq * 0.5 * (0.7 + 0.3 * variance_bias)
            split_idx = int(np.searchsorted(cum_freq, target))
            split_idx = max(1, min(len(vals_sorted)-1, split_idx))
            left_vals = [node.entries[i] for i in order[:split_idx]]
            right_vals = [node.entries[i] for i in order[split_idx:]]
        node.entries = []
        node.save()
        left = TreeNode(self.capacity, self.dir, f"{node.dir.name}L")
        right = TreeNode(self.capacity, self.dir, f"{node.dir.name}R")
        left.entries = left_vals
        right.entries = right_vals
        left.save()
        right.save()
        node.children = [left, right]

    def quantize(self, values):
        import numpy as np
        values = np.asarray(values, dtype=np.float32).flatten()
        layer_ids = np.zeros(len(values), dtype=np.int32)
        code_ids = np.zeros(len(values), dtype=np.int32)
        # Map via value_to_code, add on the fly if missing, and track lookup frequency
        for i, v in enumerate(values):
            v_f = float(v)
            if v_f not in self.value_to_code:
                self.add(v_f)
            code_ids[i] = int(self.value_to_code[v_f])
            layer_ids[i] = 0
            self.access_counts[v_f] = self.access_counts.get(v_f, 0) + 1
        return layer_ids, code_ids

    def dequantize(self, layer_ids, code_ids):
        import numpy as np
        # Reconstruct values from code ids
        vals = []
        for c in code_ids:
            idx = int(c)
            if 0 <= idx < len(self.code_to_value):
                vals.append(self.code_to_value[idx])
            else:
                # Unknown code, return 0.0
                vals.append(0.0)
        return np.array(vals, dtype=np.float32)

class TreeIndex(TreeStore):
    def __init__(self, capacity_per_node: int = 4096, dir_path: str = "index_tree"):
        super().__init__(capacity_per_node, dir_path, "index")

    def add(self, entries: dict):
        for k, v in entries.items():
            node = self._find_leaf()
            if len(node.entries) >= self.capacity:
                self._split_node(node)
                node = self._find_leaf()
            node.entries.append((k, json.dumps(v)))
            node.save()

class TreeLedger(TreeStore):
    def __init__(self, capacity_per_node: int = 10000, dir_path: str = "ledger_tree"):
        super().__init__(capacity_per_node, dir_path, "ledger")

    def append(self, entry: dict):
        node = self._find_leaf()
        if len(node.entries) >= self.capacity:
            self._split_node(node)
            node = self._find_leaf()
        node.entries.append(json.dumps(entry))
        node.save()
