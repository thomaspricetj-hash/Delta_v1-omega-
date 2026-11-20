# Full integrated training launch
# Sets all improvements together: auto-grow sliding-window memories, fractal zoning, radix sort, brachistochrone LR, pattern updates disabled for speed

$env:FILES_PER_CHUNK = 1
$env:MAX_TEXTS = 50000
$env:LAYER_DEPTH = 1
$env:OVERLAP_TEXTS = 5000
$env:MICRO_CHUNK_ROWS = 20000
$env:USE_TEXT_CACHE = 1
$env:TRAIN_STEPS = 50000
$env:DIM = 288
$env:HEADS = 9
$env:SAVE_EVERY = 20000
$env:INITIAL_BS = 8
$env:MEM_DEPTH = 6
$env:MEM_MAX_DEPTH = 64
$env:MEM_AUTO_GROW = 1
$env:MEM_WINDOW = 12
$env:SKIP_WARMUP = 1
$env:DISABLE_PATTERN_UPDATES = 1
$env:DISABLE_TALK_CHECK = 0
$env:VAL_INTERVAL = 500
$env:SELF_TALK_EVERY = 1000
$env:USE_BRACHISTOCHRONE = 1
$env:RIEMANN_ANN = 1
$env:BASE_LR = 5e-5
$env:WARMUP_STEPS = 4000

Write-Host "Starting full integrated trainer with all features enabled"
.venv\Scripts\python.exe scripts\train_weave_decoder_gpu.py
