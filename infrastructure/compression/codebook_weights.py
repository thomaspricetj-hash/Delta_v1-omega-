"""
Learned codebook compression for neural network weights
Extreme compression via symbol codebook quantization
"""
import torch
import numpy as np
from typing import Dict, Tuple
from pathlib import Path
import pickle
import math

class LearnedCodebook:
    def __init__(self, codebook_size: int = 4096):
        self.codebook_size = codebook_size
        self.codebook = None
        self.scale = 1.0
        self.zero_point = 0.0

    def fit(self, data: torch.Tensor):
        # Flatten data
        flat = data.detach().flatten().cpu().numpy().astype(np.float32)
        # Simple k-means via sklearn if available
        try:
            from sklearn.cluster import MiniBatchKMeans
            k = min(self.codebook_size, len(flat)//100)
            if k < 2:
                k = 2
            kmeans = MiniBatchKMeans(n_clusters=k, batch_size=1024, n_init=1, random_state=0)
            kmeans.fit(flat.reshape(-1,1))
            self.codebook = kmeans.cluster_centers_.flatten()
            self.codebook_size = k
        except Exception:
            # Fallback: uniform quantization
            min_v = flat.min()
            max_v = flat.max()
            self.codebook = np.linspace(min_v, max_v, self.codebook_size, dtype=np.float32)
        # Normalize for better entropy
        self.scale = float(flat.std() + 1e-9)

    def quantize(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        if self.codebook is None:
            raise RuntimeError("Codebook not fitted")
        flat = tensor.detach().flatten().cpu().numpy().astype(np.float32)
        # Find nearest codebook entry
        # Using broadcasting for simplicity - can be optimized with KDTree
        cb = self.codebook.reshape(-1,1)
        # Compute distances
        # For speed, use numpy
        diff = flat[:, None] - cb
        indices = np.argmin(np.abs(diff), axis=1).astype(np.uint8)
        # Store shape and indices
        shape = tensor.shape
        indices_tensor = torch.from_numpy(indices).view(shape)
        return indices_tensor, {'shape': shape, 'dtype': str(tensor.dtype)}

    def dequantize(self, indices: torch.Tensor, meta: Dict) -> torch.Tensor:
        cb = torch.from_numpy(self.codebook).to(indices.device)
        # Map indices to codebook values
        # indices are uint8, need to map
        flat_idx = indices.flatten().cpu().numpy()
        vals = self.codebook[flat_idx]
        tensor = torch.from_numpy(vals.astype(np.float32)).view(meta['shape'])
        return tensor.to(indices.device)


class CodebookWeightCompressor:
    def __init__(self, codebook_size: int = 4096):
        self.codebook_size = codebook_size
        self.codebooks = {}  # per tensor name

    def compress_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> Dict:
        compressed = {}
        codebook_meta = {}
        for name, tensor in state_dict.items():
            cb = LearnedCodebook(self.codebook_size)
            cb.fit(tensor)
            indices, meta = cb.quantize(tensor)
            # Store indices as bytes
            compressed[name] = {
                'indices': indices.cpu().numpy().tobytes(),
                'shape': meta['shape'],
                'dtype': str(tensor.dtype),
                'codebook': cb.codebook.tolist(),
                'scale': cb.scale
            }
            codebook_meta[name] = {
                'codebook_size': len(cb.codebook)
            }
        return {'weights': compressed, 'meta': codebook_meta}

    def decompress_state_dict(self, compressed: Dict) -> Dict[str, torch.Tensor]:
        state = {}
        weights = compressed['weights']
        for name, data in weights.items():
            # Rebuild codebook
            cb = LearnedCodebook()
            cb.codebook = np.array(data['codebook'], dtype=np.float32)
            # Rebuild indices tensor
            shape = tuple(data['shape'])
            # infer dtype size
            # Assume indices stored as uint8
            numel = int(np.prod(shape))
            indices_bytes = data['indices']
            indices = np.frombuffer(indices_bytes, dtype=np.uint8)
            indices = indices[:numel]
            indices_tensor = torch.from_numpy(indices).view(shape)
            # Dequantize
            # Simple mapping
            cb_array = torch.from_numpy(cb.codebook)
            # Map via numpy for speed
            flat_idx = indices_tensor.flatten().cpu().numpy()
            vals = cb.codebook[flat_idx]
            tensor = torch.from_numpy(vals.astype(np.float32)).view(shape)
            state[name] = tensor
        return state

    def save_compressed(self, state_dict, path: Path):
        comp = self.compress_state_dict(state_dict)
        with open(path.with_suffix('.codebook.pkl'), 'wb') as f:
            pickle.dump(comp, f)

    def load_compressed(self, path: Path):
        with open(path.with_suffix('.codebook.pkl'), 'rb') as f:
            comp = pickle.load(f)
        return self.decompress_state_dict(comp)
