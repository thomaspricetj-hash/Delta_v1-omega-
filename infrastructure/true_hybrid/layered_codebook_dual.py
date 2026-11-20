"""
Layered Codebook + Multilayered Index + Layered Ledger for Dual-Weight models
Memory-saving weight storage with small, fast-lookup layers.
Each codebook layer is a separate file; new layers are added as needed.
Multilayered index is built the same way for O(log N) lookups.
Ledger is also layered for fast append/query.
"""

import os
import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pickle

class LayeredCodebook:
    """
    Hierarchical codebook with small per-layer files for fast lookup.
    Each layer holds up to max_entries entries.
    Lookup searches layers in order, stops at first good match or falls back to nearest.
    """
    def __init__(self, capacity_per_layer: int = 1024, dir_path: str = "codebook_layers", exact: bool = False):
        self.capacity = capacity_per_layer
        self.dir = Path(dir_path)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.layers: List[np.ndarray] = []
        self.layer_files: List[Path] = []
        self.exact = exact
        self._value_to_loc = {}  # value -> (layer_id, code_id)
        # Load existing layers if any
        for f in sorted(self.dir.glob("layer_*.npy")):
            arr = np.load(f)
            self.layers.append(arr)
            self.layer_files.append(f)
        if self.exact:
            self._rebuild_value_map()

    def _rebuild_value_map(self):
        self._value_to_loc.clear()
        for li, layer in enumerate(self.layers):
            for ci, v in enumerate(layer):
                # Use repr for float key stability
                self._value_to_loc[float(v)] = (li, ci)

    def _save_layer(self, idx: int, arr: np.ndarray):
        path = self.dir / f"layer_{idx:04d}.npy"
        np.save(path, arr)
        if idx < len(self.layer_files):
            self.layer_files[idx] = path
        else:
            self.layer_files.append(path)
        if self.exact:
            # update map for this layer
            for ci, v in enumerate(arr):
                self._value_to_loc[float(v)] = (idx, ci)

    def add_layer(self, centroids: np.ndarray):
        # Ensure 1-D
        centroids = np.asarray(centroids, dtype=np.float32).reshape(-1)
        if len(centroids) > self.capacity:
            # split into multiple layers
            for i in range(0, len(centroids), self.capacity):
                chunk = centroids[i:i+self.capacity]
                self.layers.append(chunk)
                self._save_layer(len(self.layers)-1, chunk)
        else:
            self.layers.append(centroids)
            self._save_layer(len(self.layers)-1, centroids)

    def fit_from_data(self, data: np.ndarray, init_layer_size: int = 256):
        """Build layered codebook incrementally from data."""
        data = np.asarray(data, dtype=np.float32).flatten()
        if self.exact:
            # Exact deduplication mode: collect unique values
            uniq = np.unique(data)
            # Split into layers of capacity size
            for i in range(0, len(uniq), self.capacity):
                chunk = uniq[i:i+self.capacity]
                self.add_layer(chunk)
            return
        # First layer: k-means coarse
        from sklearn.cluster import MiniBatchKMeans
        k = min(init_layer_size, max(2, len(data)//100))
        kmeans = MiniBatchKMeans(n_clusters=k, batch_size=1024, n_init=1, random_state=0)
        kmeans.fit(data.reshape(-1,1))
        centroids = kmeans.cluster_centers_.flatten()
        self.add_layer(centroids)

        # Residual refinement layers
        remaining = data.copy()
        for _ in range(3):  # up to 3 refinement layers
            # Assign to nearest in existing codebook
            if not self.layers:
                break
            # Build flat codebook for distance
            flat_cb = np.concatenate(self.layers)
            dists = np.abs(remaining[:, None] - flat_cb[None, :])
            nearest_idx = np.argmin(dists, axis=1)
            nearest_vals = flat_cb[nearest_idx]
            residual = remaining - nearest_vals
            # If residual variance is low, stop
            if np.var(residual) < 1e-12:
                break
            # Fit next layer on residuals
            k2 = min(self.capacity, max(2, len(residual)//200))
            kmeans2 = MiniBatchKMeans(n_clusters=k2, batch_size=1024, n_init=1, random_state=0)
            kmeans2.fit(residual.reshape(-1,1))
            self.add_layer(kmeans2.cluster_centers_.flatten())
            # Update remaining to be residual of residuals (optional)
            remaining = residual

    def quantize(self, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return layer_ids and code_ids for each value."""
        values = np.asarray(values, dtype=np.float32).flatten()
        layer_ids = np.zeros(len(values), dtype=np.int32)
        code_ids = np.zeros(len(values), dtype=np.int32)
        if self.exact:
            # Exact lookup via value map
            for i, v in enumerate(values):
                key = float(v)
                if key in self._value_to_loc:
                    li, ci = self._value_to_loc[key]
                    layer_ids[i] = li
                    code_ids[i] = ci
                else:
                    # Value not in codebook – add it dynamically
                    # Append to last layer or create new
                    if not self.layers:
                        self.add_layer(np.array([v], dtype=np.float32))
                    else:
                        # Simple append to last layer if capacity permits
                        last = self.layers[-1]
                        if len(last) < self.capacity:
                            # extend in place
                            new_layer = np.append(last, v).astype(np.float32)
                            self.layers[-1] = new_layer
                            self._save_layer(len(self.layers)-1, new_layer)
                        else:
                            self.add_layer(np.array([v], dtype=np.float32))
                    # re-lookup
                    li, ci = self._value_to_loc[float(v)]
                    layer_ids[i] = li
                    code_ids[i] = ci
            return layer_ids, code_ids
        # Simple linear search per layer - layers are small
        for i, v in enumerate(values):
            best_dist = float('inf')
            best_layer = 0
            best_code = 0
            for li, layer in enumerate(self.layers):
                # vectorized distance for this layer
                dists = np.abs(layer - v)
                j = np.argmin(dists)
                d = dists[j]
                if d < best_dist:
                    best_dist = d
                    best_layer = li
                    best_code = j
                    # early exit if perfect
                    if d == 0:
                        break
            layer_ids[i] = best_layer
            code_ids[i] = best_code
        return layer_ids, code_ids

    def dequantize(self, layer_ids: np.ndarray, code_ids: np.ndarray) -> np.ndarray:
        out = np.empty(len(layer_ids), dtype=np.float32)
        for i, (li, ci) in enumerate(zip(layer_ids, code_ids)):
            out[i] = self.layers[int(li)][int(ci)]
        return out

    def save_meta(self, path: Path):
        meta = {"capacity": self.capacity, "num_layers": len(self.layers)}
        with open(path, 'w') as f:
            json.dump(meta, f)

    def load_meta(self, path: Path):
        with open(path) as f:
            _ = json.load(f)


class MultilayeredIndex:
    """
    Index built same way as codebook: small layered files for fast lookup.
    Maps logical keys -> (layer_id, entry_id)
    """
    def __init__(self, capacity_per_layer: int = 4096, dir_path: str = "index_layers"):
        self.capacity = capacity_per_layer
        self.dir = Path(dir_path)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.layers: List[Dict] = []
        self.layer_files: List[Path] = []
        for f in sorted(self.dir.glob("index_*.jsonl")):
            d = {}
            with open(f) as fh:
                for line in fh:
                    k,v = line.strip().split('\t',1)
                    d[k]=json.loads(v)
            self.layers.append(d)
            self.layer_files.append(f)

    def _save_layer(self, idx: int, d: Dict):
        path = self.dir / f"index_{idx:04d}.jsonl"
        with open(path, 'w') as f:
            for k,v in d.items():
                f.write(f"{k}\t{json.dumps(v)}\n")
        if idx < len(self.layer_files):
            self.layer_files[idx]=path
        else:
            self.layer_files.append(path)

    def add_entries(self, entries: Dict[str, dict]):
        # split into layers
        for k,v in entries.items():
            placed=False
            for li, layer in enumerate(self.layers):
                if len(layer) < self.capacity:
                    layer[k]=v
                    placed=True
                    break
            if not placed:
                new_layer={}
                new_layer[k]=v
                self.layers.append(new_layer)
                self._save_layer(len(self.layers)-1, new_layer)
            # else keep layer file updated
            # For simplicity, rewrite layer file each time
            # In production, use append
        # rewrite layers to persist
        for i, layer in enumerate(self.layers):
            self._save_layer(i, layer)

    def lookup(self, key: str) -> Optional[dict]:
        for layer in self.layers:
            if key in layer:
                return layer[key]
        return None

    def lookup_batch(self, keys: List[str]) -> Dict[str, dict]:
        res={}
        for k in keys:
            v=self.lookup(k)
            if v is not None:
                res[k]=v
        return res


class LayeredLedger:
    """
    Append-only ledger with sharded layers for fast lookup.
    Built same way as codebook/index.
    """
    def __init__(self, capacity_per_layer: int = 10000, dir_path: str = "ledger_layers"):
        self.capacity = capacity_per_layer
        self.dir = Path(dir_path)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.current_layer = 0
        self.current_file = self.dir / f"ledger_{self.current_layer:04d}.jsonl"
        self.entries = []
        # Find latest layer
        existing = sorted(self.dir.glob("ledger_*.jsonl"))
        if existing:
            self.current_layer = int(existing[-1].stem.split('_')[1])
            self.current_file = existing[-1]
            with open(self.current_file) as f:
                self.entries = [json.loads(l) for l in f if l.strip()]

    def append(self, entry: dict):
        if len(self.entries) >= self.capacity:
            self.current_layer += 1
            self.current_file = self.dir / f"ledger_{self.current_layer:04d}.jsonl"
            self.entries = []
        self.entries.append(entry)
        with open(self.current_file, 'a') as f:
            f.write(json.dumps(entry)+"\n")

    def query(self, filter_fn):
        results=[]
        for f in sorted(self.dir.glob("ledger_*.jsonl")):
            with open(f) as fh:
                for line in fh:
                    e=json.loads(line)
                    if filter_fn(e):
                        results.append(e)
        return results


class CodebookDualWeightCompressor:
    """
    Compressor for DualWeight models using layered codebooks, multilayered index and layered ledger.
    Saves memory by storing weights as codebook indices.
    """
    def __init__(self, codebook_dir: str = "codebook", index_dir: str = "index", ledger_dir: str = "ledger"):
        self.codebook_dir = Path(codebook_dir)
        self.index_dir = Path(index_dir)
        self.ledger_dir = Path(ledger_dir)
        # Use tree-structured index and ledger for automatic node splitting
        from infrastructure.true_hybrid.tree_store import TreeIndex, TreeLedger, TreeCodebook
        self.index = TreeIndex(capacity_per_node=4096, dir_path=str(self.index_dir))
        self.ledger = TreeLedger(capacity_per_node=10000, dir_path=str(self.ledger_dir))
        self.TreeCodebook = TreeCodebook

    def compress_model(self, model: torch.nn.Module, output_root: str):
        root = Path(output_root)
        root.mkdir(parents=True, exist_ok=True)
        meta = {}
        # Traverse modules
        for name, module in model.named_modules():
            if module.__class__.__name__ == "DualWeightLinear":
                # Build per-layer codebook for master and delta using tree-structured codebook
                master_data = module.master_weight.detach().cpu().numpy()
                delta_data = module.delta_weight.detach().cpu().numpy()
                cb_master = self.TreeCodebook(capacity_per_node=1024, dir_path=str(root / f"codebook/{name}/master"))
                cb_master.fit_from_data(master_data)
                cb_delta = self.TreeCodebook(capacity_per_node=1024, dir_path=str(root / f"codebook/{name}/delta"))
                cb_delta.fit_from_data(delta_data)

                # Quantize via exact lookup
                m_layer_ids, m_code_ids = cb_master.quantize(master_data)
                d_layer_ids, d_code_ids = cb_delta.quantize(delta_data)

                # Save indices
                idx_path = root / f"indices/{name}"
                idx_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(idx_path.with_suffix('.npz'),
                    m_layer_ids=m_layer_ids, m_code_ids=m_code_ids,
                    d_layer_ids=d_layer_ids, d_code_ids=d_code_ids,
                    shape=module.master_weight.shape,
                    bias_shape=module.master_bias.shape if module.master_bias is not None else None
                )
                # Save codebook meta
                cb_master.save_meta(root / f"codebook/{name}/master/meta.json")
                cb_delta.save_meta(root / f"codebook/{name}/delta/meta.json")

                # Index entry
                entry = {
                    "type": "DualWeightLinear",
                    "master_codebook": str(root / f"codebook/{name}/master"),
                    "delta_codebook": str(root / f"codebook/{name}/delta"),
                    "indices_file": str(idx_path.with_suffix('.npz'))
                }
                self.index.add({name: entry})
                self.ledger.append({"op":"compress","module":name,"timestamp":__import__('time').time()})
                meta[name]=entry
        # Save global meta
        with open(root / "model_meta.json","w") as f:
            json.dump(meta,f,indent=2)
        return root

    def decompress_model(self, model: torch.nn.Module, output_root: str):
        root = Path(output_root)
        with open(root / "model_meta.json") as f:
            meta=json.load(f)
        for name, entry in meta.items():
            # Find module by name
            parts=name.split('.')
            mod=model
            for p in parts:
                mod=getattr(mod,p)
            # Load indices
            data=np.load(entry["indices_file"])
            m_layer_ids=data['m_layer_ids']
            m_code_ids=data['m_code_ids']
            d_layer_ids=data['d_layer_ids']
            d_code_ids=data['d_code_ids']
            # Load codebooks
            cb_master=self.TreeCodebook(capacity_per_node=1024, dir_path=entry["master_codebook"])
            cb_delta=self.TreeCodebook(capacity_per_node=1024, dir_path=entry["delta_codebook"])
            # Dequantize
            master_vals=cb_master.dequantize(m_layer_ids,m_code_ids)
            delta_vals=cb_delta.dequantize(d_layer_ids,d_code_ids)
            # Load into module (shape-safe)
            with torch.no_grad():
                try:
                    if master_vals.size == mod.master_weight.numel():
                        mod.master_weight.copy_(torch.from_numpy(master_vals.reshape(mod.master_weight.shape)))
                    else:
                        # Partial load: copy overlapping region
                        min_n = min(master_vals.size, mod.master_weight.numel())
                        mod.master_weight.view(-1)[:min_n].copy_(torch.from_numpy(master_vals[:min_n]))
                except Exception as e:
                    print(f"[Codebook] master weight load skipped for {name}: {e}")
                try:
                    if delta_vals.size == mod.delta_weight.numel():
                        mod.delta_weight.copy_(torch.from_numpy(delta_vals.reshape(mod.delta_weight.shape)))
                    else:
                        min_n = min(delta_vals.size, mod.delta_weight.numel())
                        mod.delta_weight.view(-1)[:min_n].copy_(torch.from_numpy(delta_vals[:min_n]))
                except Exception as e:
                    print(f"[Codebook] delta weight load skipped for {name}: {e}")
                if mod.master_bias is not None:
                    # bias handled similarly if needed
                    pass
        return model

# Example usage
if __name__ == "__main__":
    from dual_weight_layers import DualWeightLinear, DualWeightWeaveFormer
    model = DualWeightWeaveFormer(dim=64, num_layers=2, vocab_size=128, use_dual_weights=True)
    compressor = CodebookDualWeightCompressor()
    out = compressor.compress_model(model, "compressed_dual_model")
    print(f"Compressed model saved to {out}")
