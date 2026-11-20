"""
Example: Using layered codebook for DualWeight memory saving
"""
import torch
from dual_weight_layers import DualWeightWeaveFormer
from layered_codebook_dual import CodebookDualWeightCompressor

def demo():
    print("Creating small DualWeight model...")
    model = DualWeightWeaveFormer(dim=128, num_layers=4, vocab_size=256, use_dual_weights=True)
    # Dummy forward
    x = torch.randint(0,256,(2,16))
    _ = model(x)
    print("Model created, compressing...")
    compressor = CodebookDualWeightCompressor()
    out_dir = "compressed_dual_demo"
    compressor.compress_model(model, out_dir)
    print(f"Compressed to {out_dir}")
    print("Rebuilding model and decompressing...")
    model2 = DualWeightWeaveFormer(dim=128, num_layers=4, vocab_size=256, use_dual_weights=True)
    compressor.decompress_model(model2, out_dir)
    # Verify shapes
    print("Decompression complete. Model ready for inference.")
    _ = model2(x)
    print("Forward pass successful.")

if __name__ == "__main__":
    demo()
