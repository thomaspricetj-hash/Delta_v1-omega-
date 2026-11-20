# Layered Codebook for Dual-Weight Models

## Overview
Dual-Weight layers store master + delta weights. To save memory, weights are stored via an exact, lossless layered codebook/deduplication system – no quantization. Unique weight values are kept in small per-layer codebook files and referenced by indices.

* **LayeredCodebook** – codebook vectors are stored in small per-layer `.npy` files. New layers are added when capacity is reached, keeping each layer small for fast lookup.
* **MultilayeredIndex** – maps module names → codebook locations. Index itself is layered (`.jsonl` shards) for fast lookup.
* **LayeredLedger** – append-only ledger with sharded layers for audit / provenance.

## Files
* `layered_codebook_dual.py` – core classes: `LayeredCodebook`, `MultilayeredIndex`, `LayeredLedger`, `CodebookDualWeightCompressor`
* `codebook_lazy_dual.py` – `CodebookDualWeightLinear` with lazy dequantization, `make_codebook_dual_weight_layer`
* `convert_to_codebook.py` – model-wide conversion to codebook lazy mode
* `codebook_dual_example.py` – demo of compress/decompress

## Usage
```python
from dual_weight_layers import DualWeightWeaveFormer
from layered_codebook_dual import CodebookDualWeightCompressor

model = DualWeightWeaveFormer(dim=256, num_layers=32, vocab_size=512)
compressor = CodebookDualWeightCompressor()
compressor.compress_model(model, "compressed_model")
```

Compression stores each `DualWeightLinear` master/delta weights as codebook indices. Decompression rebuilds the weights on demand.

### Lazy inference mode
```python
from convert_to_codebook import convert_model_to_codebook
model = convert_model_to_codebook(model, codebook_capacity=1024)
# Weights stay as indices + small layered codebooks; forward dequantizes lazily and caches
```

Memory savings: indices are 32-bit per weight vs 32-bit float, plus codebook sharing across layers. With layered codebooks, lookup stays fast because each layer is small.

## Extending
* Set `capacity_per_layer` to tune lookup speed vs file count.
* Ledger can be used for training provenance: each compress/decompress appends an entry.
* Index can be queried for module metadata without loading weights.
