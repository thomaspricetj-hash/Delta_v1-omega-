"""
Codebook checkpoint helpers for DualWeight models
Saves model weights as layered exact codebooks instead of raw tensors.
"""

from pathlib import Path
from infrastructure.true_hybrid.layered_codebook_dual import CodebookDualWeightCompressor
import torch

def save_model_codebook(model, save_dir, base_name="model_codebook", extra_state=None):
    """
    Save model using layered exact codebook compression.
    Returns path to metadata file.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    compressor = CodebookDualWeightCompressor(
        codebook_dir=str(save_dir / "codebook"),
        index_dir=str(save_dir / "index"),
        ledger_dir=str(save_dir / "ledger")
    )
    out_root = save_dir / base_name
    compressor.compress_model(model, str(out_root))
    # Save metadata
    meta = {"codebook_root": str(out_root)}
    meta_path = save_dir / f"{base_name}_meta.pt"
    torch.save(meta, meta_path)
    # Save extra training state if provided
    if extra_state is not None:
        state_path = save_dir / f"{base_name}_state.pt"
        torch.save(extra_state, state_path)
    return out_root

def load_model_codebook(model, load_dir, base_name="model_codebook"):
    """
    Load model weights from codebook representation.
    Returns (model, extra_state) where extra_state may be None.
    """
    load_dir = Path(load_dir)
    meta_path = load_dir / f"{base_name}_meta.pt"
    if not meta_path.exists():
        raise FileNotFoundError(f"Codebook meta not found at {meta_path}")
    meta = torch.load(meta_path, map_location='cpu')
    codebook_root = meta["codebook_root"]
    compressor = CodebookDualWeightCompressor(
        codebook_dir=str(Path(codebook_root).parent / "codebook"),
        index_dir=str(Path(codebook_root).parent / "index"),
        ledger_dir=str(Path(codebook_root).parent / "ledger")
    )
    compressor.decompress_model(model, codebook_root)
    # Load extra training state if present
    state_path = load_dir / f"{base_name}_state.pt"
    extra_state = None
    if state_path.exists():
        extra_state = torch.load(state_path, map_location='cpu')
    return model, extra_state
