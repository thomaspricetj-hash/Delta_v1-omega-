#!/usr/bin/env python3
"""
Export predictor savings to Prometheus gauges
predictor_matrix_multiply_hits_total and predictor_matrix_multiply_saved_ms_total
"""
import json
from pathlib import Path
BASE = Path(__file__).resolve().parents[1]
SAVINGS_PATH = BASE / "runtime" / "logs" / "predictor_savings.jsonl"
PROM_PATH = BASE / "runtime" / "logs" / "predictor_savings.prom"

hits_total = 0
saved_ms_total = 0.0
misses = 0
if SAVINGS_PATH.exists():
    with open(SAVINGS_PATH,'r',encoding='utf-8') as f:
        for line in f:
            try:
                e = json.loads(line)
                if e.get('hit'):
                    hits_total += 1
                    saved_ms_total += float(e.get('est_saved_ms',0))
                else:
                    misses += 1
            except Exception:
                pass

with open(PROM_PATH,'w',encoding='utf-8') as f:
    f.write('# HELP predictor_matrix_multiply_hits_total Total cache hits\n')
    f.write('# TYPE predictor_matrix_multiply_hits_total counter\n')
    f.write(f'predictor_matrix_multiply_hits_total {hits_total}\n')
    f.write('# HELP predictor_matrix_multiply_saved_ms_total Estimated ms saved\n')
    f.write('# TYPE predictor_matrix_multiply_saved_ms_total counter\n')
    f.write(f'predictor_matrix_multiply_saved_ms_total {saved_ms_total}\n')
    f.write('# HELP predictor_matrix_multiply_misses_total Total cache misses\n')
    f.write('# TYPE predictor_matrix_multiply_misses_total counter\n')
    f.write(f'predictor_matrix_multiply_misses_total {misses}\n')

print(f"Exported hits={hits_total}, saved_ms={saved_ms_total:.2f}, misses={misses}")
