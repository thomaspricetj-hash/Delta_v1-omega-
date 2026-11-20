"""
Auto layered chunk curriculum with sliding window.
Progresses through chunks and layer depths, reusing cache.
"""
import os
import sys
from pathlib import Path
import subprocess
import glob

BASE = Path(__file__).resolve().parent.parent
TRAINER = str(BASE / "scripts" / "train_weave_decoder_gpu.py")
PYTHON = str(BASE / ".venv" / "Scripts" / "python.exe")

def run_trainer(env):
    # Merge env with current
    merged = os.environ.copy()
    merged.update(env)
    print(f"\n=== Running trainer with CHUNK_INDEX={env.get('CHUNK_INDEX')}, LAYER_DEPTH={env.get('LAYER_DEPTH')} ===")
    result = subprocess.run([PYTHON, TRAINER], env=merged, cwd=str(BASE))
    return result.returncode

def main():
    data_dir = BASE / "trainingData"
    all_files = sorted(glob.glob(str(data_dir / "*.parquet")))
    total_files = len(all_files)
    files_per_chunk = int(os.environ.get('FILES_PER_CHUNK', '1'))
    max_texts = os.environ.get('MAX_TEXTS', '50000')
    overlap_texts = os.environ.get('OVERLAP_TEXTS', '5000')
    use_cache = os.environ.get('USE_TEXT_CACHE', '1')
    train_steps = os.environ.get('TRAIN_STEPS_PER_CHUNK', '50000')
    dim = os.environ.get('DIM', '288')
    heads = os.environ.get('HEADS', '9')
    save_every = os.environ.get('SAVE_EVERY', '20000')

    total_chunks = (total_files + files_per_chunk - 1) // files_per_chunk
    layer_depths = [1, 2, 3]

    for chunk_idx in range(total_chunks):
        for layer_depth in layer_depths:
            env = {
                'CHUNK_INDEX': str(chunk_idx),
                'FILES_PER_CHUNK': str(files_per_chunk),
                'LAYER_DEPTH': str(layer_depth),
                'MAX_TEXTS': max_texts,
                'OVERLAP_TEXTS': overlap_texts,
                'USE_TEXT_CACHE': use_cache,
                'TRAIN_STEPS': train_steps,
                'DIM': dim,
                'HEADS': heads,
                'SAVE_EVERY': save_every,
                'MICRO_CHUNK_ROWS': os.environ.get('MICRO_CHUNK_ROWS', '20000'),
            }
            rc = run_trainer(env)
            if rc != 0:
                print(f"[WARN] Trainer exited with code {rc}, continuing to next layer/chunk")
            # Simple plateau check could be added here by parsing logs
            # For now, continue to next layer depth
        print(f"[INFO] Finished chunk {chunk_idx}, moving to next chunk")
    print("[DONE] Layered curriculum complete")

if __name__ == "__main__":
    main()
