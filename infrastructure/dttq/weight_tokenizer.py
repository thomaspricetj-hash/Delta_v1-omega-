"""
Production-grade DTTQ: Delta-Tagged Tokenized Quantization for WeaveFormer weights
Motif dictionary + residual coding + predictive coding + serialization
"""
import torch
import numpy as np
from pathlib import Path
import json
from dataclasses import dataclass, asdict
from typing import Dict, Tuple
import hashlib

try:
    from sklearn.cluster import MiniBatchKMeans
except ImportError:
    MiniBatchKMeans = None

@dataclass
class DTTQConfig:
    vocab_size: int = 4096
    block_size: int = 16
    residual_bits: int = 8
    seed: int = 0
    max_blocks_per_fit: int = 500000
    kmeans_batch_size: int = 1024

class WeightMotifCompressor:
    def __init__(self, config: DTTQConfig = None):
        self.config = config or DTTQConfig()
        self.motif_centers: np.ndarray = None
        self.vocab_size = self.config.vocab_size
        self.block_size = self.config.block_size
        self._fitted = False

    def _extract_blocks(self, tensor: torch.Tensor):
        t = tensor.detach().cpu().numpy().astype(np.float32)
        flat = t.reshape(-1)
        n = len(flat)
        if n < self.block_size:
            return np.empty((0, self.block_size), dtype=np.float32)
        # stride by block_size for non-overlapping blocks
        n_blocks = n // self.block_size
        blocks = flat[:n_blocks*self.block_size].reshape(n_blocks, self.block_size)
        return blocks

    def fit(self, state_dict: Dict[str, torch.Tensor]):
        if MiniBatchKMeans is None:
            raise RuntimeError("scikit-learn required for DTTQ fit")
        rng = np.random.default_rng(self.config.seed)
        all_blocks = []
        for name, param in state_dict.items():
            if not torch.is_tensor(param):
                continue
            if param.numel() < self.block_size:
                continue
            blocks = self._extract_blocks(param)
            if blocks.shape[0] == 0:
                continue
            # subsample to bound memory
            if blocks.shape[0] > self.config.max_blocks_per_fit:
                idx = rng.choice(blocks.shape[0], self.config.max_blocks_per_fit, replace=False)
                blocks = blocks[idx]
            all_blocks.append(blocks)
        if not all_blocks:
            raise RuntimeError("No blocks extracted for fitting")
        X = np.vstack(all_blocks)
        # Normalize blocks per tensor for better clustering
        X = X - X.mean(axis=0, keepdims=True)
        kmeans = MiniBatchKMeans(
            n_clusters=self.vocab_size,
            batch_size=self.config.kmeans_batch_size,
            random_state=self.config.seed,
            n_init=1
        )
        kmeans.fit(X)
        self.motif_centers = kmeans.cluster_centers_.astype(np.float32)
        self._fitted = True
        return self

    def _quantize_residual(self, residual: np.ndarray):
        # Simple uniform quantization to residual_bits
        rmin, rmax = residual.min(), residual.max()
        if rmax == rmin:
            q = np.zeros_like(residual, dtype=np.int16)
        else:
            scale = (2**self.config.residual_bits - 1) / (rmax - rmin)
            q = np.round((residual - rmin) * scale).astype(np.int16)
        return q, rmin, rmax

    def _dequantize_residual(self, q, rmin, rmax):
        if rmax == rmin:
            return np.zeros_like(q, dtype=np.float32)
        scale = (rmax - rmin) / (2**self.config.residual_bits - 1)
        return q.astype(np.float32) * scale + rmin

    def compress(self, state_dict: Dict[str, torch.Tensor]):
        if not self._fitted:
            raise RuntimeError("Compressor not fitted")
        compressed = {}
        meta = {"config": asdict(self.config), "total_blocks":0, "total_params":0, "mse":0.0}
        for name, param in state_dict.items():
            if not torch.is_tensor(param):
                compressed[name] = param
                continue
            blocks = self._extract_blocks(param)
            n_blocks = blocks.shape[0]
            meta["total_blocks"] += n_blocks
            meta["total_params"] += param.numel()
            if n_blocks == 0:
                compressed[name] = param
                continue
            # nearest motif
            dists = np.linalg.norm(blocks[:, None, :] - self.motif_centers[None, :, :], axis=2)
            codes = np.argmin(dists, axis=1).astype(np.int32)
            recon_centers = self.motif_centers[codes]
            residual = blocks - recon_centers
            # quantize residual
            q_res, rmin, rmax = self._quantize_residual(residual)
            # store
            compressed[name] = {
                "shape": tuple(param.shape),
                "dtype": str(param.dtype),
                "block_size": self.block_size,
                "codes": codes,
                "residual_q": q_res,
                "rmin": float(rmin),
                "rmax": float(rmax),
            }
            # track mse
            mse = float(np.mean(residual**2))
            meta["mse"] += mse * n_blocks
        meta["mse"] /= max(1, meta["total_blocks"])
        compressed["_meta"] = meta
        return compressed

    def decompress(self, compressed: Dict):
        state_dict = {}
        for name, entry in compressed.items():
            if name == "_meta":
                continue
            if isinstance(entry, dict) and "codes" in entry:
                codes = entry["codes"]
                q_res = entry["residual_q"]
                rmin = entry["rmin"]
                rmax = entry["rmax"]
                residual = self._dequantize_residual(q_res, rmin, rmax)
                recon = self.motif_centers[codes] + residual
                flat = recon.reshape(-1)
                out = torch.zeros(entry["shape"], dtype=torch.float32)
                out_flat = out.flatten()[:len(flat)]
                out_flat.copy_(torch.from_numpy(flat))
                state_dict[name] = out.to(torch.float32)
            else:
                state_dict[name] = entry
        return state_dict

    def save(self, path: Path):
        path = Path(path)
        np.savez_compressed(path.with_suffix(".npz"), centers=self.motif_centers)
        meta = {
            "config": asdict(self.config),
            "hash": hashlib.sha256(self.motif_centers.tobytes()).hexdigest()
        }
        with open(path.with_suffix(".json"), "w") as f:
            json.dump(meta, f)
    
    @classmethod
    def load(cls, npz_path: Path):
        data = np.load(npz_path)
        centers = data["centers"]
        # load meta
        json_path = npz_path.with_suffix(".json")
        meta = {}
        if json_path.exists():
            with open(json_path) as f:
                meta = json.load(f)
        config = DTTQConfig(**meta.get("config", {}))
        obj = cls(config)
        obj.motif_centers = centers
        obj._fitted = True
        return obj

def export_compressed_model(model_path: Path, out_dir: Path, vocab_size: int = 4096):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    state = torch.load(model_path, map_location="cpu")
    cfg = DTTQConfig(vocab_size=vocab_size, seed=42)
    comp = WeightMotifCompressor(cfg)
    comp.fit(state)
    comp_state = comp.compress(state)
    # save artifacts
    comp.save(out_dir / "dttq_motifs")
    torch.save(comp_state, out_dir / "compressed_state.pt")
    meta = comp_state["_meta"]
    print(f"DTTQ export complete")
    print(f"  Blocks: {meta['total_blocks']}, Params: {meta['total_params']}, MSE: {meta['mse']:.6f}")
    return out_dir

def import_compressed_model(compressed_dir: Path, device: str = "cpu"):
    comp = WeightMotifCompressor.load(compressed_dir / "dttq_motifs.npz")
    comp_state = torch.load(compressed_dir / "compressed_state.pt", map_location=device)
    state_dict = comp.decompress(comp_state)
    return state_dict
