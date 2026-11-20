#!/usr/bin/env python3
"""
Per-op accuracy EMA validator for matrix_multiply_predictor against math_synthetic.jsonl
Writes EMA metrics to runtime/logs/predictor_ema.jsonl and Prometheus gauges
"""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from infrastructure.predictors.matrix_multiply_predictor import predict as predictor_predict
from infrastructure.predictors.matrix_multiply_predictor import _PREDICTOR_STATS

BASE = Path(__file__).resolve().parents[1]
LOG_PATH = BASE / "runtime" / "logs" / "predictor_ema.jsonl"
SYNTH_PATH = BASE / "runtime" / "logs" / "math_synthetic.jsonl"
PROM_PATH = BASE / "runtime" / "logs" / "predictor_metrics.prom"

EMA_ALPHA = 0.1
# Load existing EMA state
ema_state = {}
if LOG_PATH.exists():
    try:
        with open(LOG_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line)
                op = obj.get('op')
                if op and 'ema' in obj:
                    ema_state[op] = obj['ema']
    except Exception:
        pass

def parse_answer(val):
    try:
        # Handle float/int string
        if '.' in str(val):
            return float(val)
        return int(val)
    except Exception:
        return None

def evaluate():
    total = 0
    correct = 0
    op_counts = {}
    op_correct = {}
    if not SYNTH_PATH.exists():
        print("No synthetic file")
        return
    with open(SYNTH_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                obj = json.loads(line)
                q = obj.get('question','')
                expected = parse_answer(obj.get('answer'))
                op = obj.get('op','?')
                # Run predictor
                pred = predictor_predict(q)
                if not pred:
                    # count as miss
                    total += 1
                    op_counts[op] = op_counts.get(op,0)+1
                    continue
                pred_val = parse_answer(pred.get('response'))
                is_correct = False
                if pred_val is not None and expected is not None:
                    # tolerance for floats
                    if isinstance(expected, float):
                        is_correct = abs(pred_val - expected) < 1e-6
                    else:
                        is_correct = pred_val == expected
                total += 1
                op_counts[op] = op_counts.get(op,0)+1
                op_correct[op] = op_correct.get(op,0) + (1 if is_correct else 0)
                correct += 1 if is_correct else 0
            except Exception:
                continue
    # Update EMA per op
    results = []
    for op in ['+','-','*','/']:
        cnt = op_counts.get(op,0)
        cor = op_correct.get(op,0)
        acc = cor / cnt if cnt>0 else 0.0
        prev_ema = ema_state.get(op, acc)
        ema = EMA_ALPHA * acc + (1-EMA_ALPHA) * prev_ema
        ema_state[op] = ema
        results.append({
            'ts': __import__('time').time(),
            'op': op,
            'count': cnt,
            'correct': cor,
            'accuracy': round(acc,4),
            'ema': round(ema,4),
            'predictor_confidence': _PREDICTOR_STATS.get(op,{}).get('confidence',0.85)
        })
    # Write log
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, 'a', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r)+'\n')
    # Prometheus gauges
    with open(PROM_PATH,'w',encoding='utf-8') as f:
        f.write('# HELP predictor_accuracy_ema Per-op accuracy EMA\n')
        f.write('# TYPE predictor_accuracy_ema gauge\n')
        for r in results:
            f.write(f'predictor_accuracy_ema{{op="{r["op"]}"}} {r["ema"]}\n')
        f.write('# HELP predictor_accuracy_current Per-op current accuracy\n')
        f.write('# TYPE predictor_accuracy_current gauge\n')
        for r in results:
            f.write(f'predictor_accuracy_current{{op="{r["op"]}"}} {r["accuracy"]}\n')
        f.write('# HELP predictor_confidence Current predictor confidence\n')
        f.write('# TYPE predictor_confidence gauge\n')
        for r in results:
            f.write(f'predictor_confidence{{op="{r["op"]}"}} {r["predictor_confidence"]}\n')
    overall_acc = correct/total if total>0 else 0
    print(f"Evaluated {total} samples, overall accuracy {overall_acc:.4f}")
    for r in results:
        print(r)

if __name__ == '__main__':
    evaluate()
