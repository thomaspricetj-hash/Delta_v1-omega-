# delta_v1 omega - TRANSLATION_OS_V1 Share Package

## Contents
* `infrastructure/` - model architecture, layers, tokenizer, data, optimization
* `scripts/` - training and health check scripts
* `runtime/models/` - trained checkpoint `weave_decoder_main.pt`
* `runtime/token_codebook/` - hierarchical codebook
* `runtime/token_dictionary.json` - word dictionary for tokenizer

## Quick start
```powershell
.venv\Scripts\python.exe scripts\train_weave_decoder_gpu.py
```

Recommended env vars:
```
TRAIN_STEPS=50000
SKIP_WARMUP=0
DIM=288
HEADS=9
INITIAL_SEQ=512
INITIAL_BS=4
ATTN_WINDOW_BASE=256
ATTN_WINDOW_GROWTH=4
ROTARY_BASE=10000
ATTN_QKV_BIAS=1
FAST_MODE=0
```

Health check:
```powershell
PYTHONIOENCODING=utf-8 .venv\Scripts\python.exe scripts\run_health_checks.py
```

Kid-learning curriculum, Pro SwiGLU FFN, ALiBi, LayerScale, sliding-window attention with global heads are all on by default.

## Training data
The trainer expects Parquet files in `trainingData/` relative to the project root.

Place your corpus as `trainingData/*.parquet` with at least one text column. Example layout:
```
TRANSLATION_OS_V1_SHARE/
  trainingData/
    part-00000.parquet
    part-00001.parquet
```

On first run the trainer will build a text cache in `runtime/cache/` and pre-build the hierarchical codebook from a sample of the parquet data. You can limit data for a quick test:
```powershell
set MAX_TEXTS=5000
set CHUNK_INDEX=0
.venv\Scripts\python.exe scripts\train_weave_decoder_gpu.py
```

If your data is elsewhere, edit the line `data_dir = BASE / "trainingData"` in `scripts/train_weave_decoder_gpu.py` to point to your folder, or copy your parquet files into `trainingData/`.

## Start training
```powershell
.venv\Scripts\python.exe scripts\train_weave_decoder_gpu.py
```

Resume from checkpoint automatically if `runtime_models/weave_decoder_main.pt` exists. Use `TRAIN_STEPS`, `SKIP_WARMUP`, `DIM`, `HEADS`, etc. to control the run.
