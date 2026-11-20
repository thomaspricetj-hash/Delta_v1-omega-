"""
Progress report every 30 min for unlimited training
Prints avg grade, stability, repeat counts, conversation sample
"""
import json
import time
from pathlib import Path
from collections import deque

BASE = Path('C:/Users/thoma/.lmstudio/apps/bionic/projects/d49037d8-47f4-5808-9028-c707de117f8f/workspace/TRANSLATION_OS_V1')
LOGS = BASE / 'runtime' / 'logs'
FACTS = BASE / 'runtime' / 'facts'

def read_last_lines(path, n=20):
    if not path.exists():
        return []
    lines = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            lines.append(line)
    return lines[-n:]

def avg_grade():
    ledger = LOGS / 'third_brain_ledger.jsonl'
    grades = []
    for line in read_last_lines(ledger, 100):
        try:
            obj = json.loads(line)
            grades.append(obj.get('avg_grade',0))
        except Exception:
            pass
    return sum(grades)/len(grades) if grades else 0

def stability():
    conv = LOGS / 'predictor_convergence.jsonl'
    stabs = []
    for line in read_last_lines(conv, 100):
        try:
            obj = json.loads(line)
            # stability is nested under past.stability, fallback to convergence_score
            stab = obj.get('stability')
            if stab is None:
                past = obj.get('past') or {}
                stab = past.get('stability')
            if stab is None:
                stab = obj.get('convergence_score', 0)
            stabs.append(stab)
        except Exception:
            pass
    return sum(stabs)/len(stabs) if stabs else 0

def repeat_counts():
    delta_path = FACTS / 'fact_deltas.jsonl'
    counts = {}
    for line in read_last_lines(delta_path, 500):
        try:
            obj = json.loads(line)
            key = f"{obj.get('subject')}::{obj.get('predicate')}"
            counts[key] = counts.get(key,0)+1
        except Exception:
            pass
    # top 5
    top = sorted(counts.items(), key=lambda x:x[1], reverse=True)[:5]
    return top

def conversation_sample():
    # simple sample from diagnostics
    diag = LOGS / 'orchestrator_diagnostics.jsonl'
    samples = []
    for line in read_last_lines(diag, 50):
        try:
            obj = json.loads(line)
            q = obj.get('query_original')
            r = obj.get('response')
            src = obj.get('response_source')
            if q and r:
                samples.append((q, r, src))
        except Exception:
            pass
    return samples[-3:]

def report():
    print('='*70)
    print(f"Progress report {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print('='*70)
    print(f"Avg Grade (last 100): {avg_grade():.4f}")
    print(f"Avg Stability (last 100): {stability():.4f}")
    print('Top repeat facts:')
    for k,v in repeat_counts():
        print(f"  {k}: {v} repeats")
    print('Recent conversation samples:')
    for q,r,src in conversation_sample():
        print(f"  [{src}] Q: {q[:80]}")
        print(f"       A: {r[:120]}")
    print('='*70)

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1]=='loop':
        while True:
            report()
            time.sleep(1800)  # 30 min
    else:
        report()
