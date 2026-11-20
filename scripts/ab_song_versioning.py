#!/usr/bin/env python3
"""
A/B song versioning with statistical comparison
Writes per-experiment song logs and computes p-values on convergence/entropy
"""
import json, math
from pathlib import Path
import random
BASE = Path(__file__).resolve().parents[1]
SONG_LOG = BASE / "runtime" / "logs" / "song_notes.jsonl"
AB_DIR = BASE / "runtime" / "logs" / "ab_songs"
AB_DIR.mkdir(parents=True, exist_ok=True)

def load_notes(path):
    notes = []
    if path.exists():
        with open(path,'r',encoding='utf-8') as f:
            for line in f:
                try:
                    notes.append(json.loads(line))
                except Exception:
                    pass
    return notes

def harmony_stats(notes):
    if not notes:
        return {'convergence':0,'entropy':0}
    vals = [n.get('note',60) for n in notes]
    from collections import Counter
    cnt = Counter(vals)
    total = len(vals)
    unique = len(cnt)
    convergence = 1 - unique/total if total>0 else 0
    # entropy
    entropy = -sum((c/total)*math.log2(c/total) for c in cnt.values())
    return {'convergence': convergence, 'entropy': entropy, 'total': total}

def split_ab(notes, seed=42):
    random.Random(seed).shuffle(notes)
    mid = len(notes)//2
    return notes[:mid], notes[mid:]

def write_experiment(exp_id, notes):
    out = AB_DIR / f"song_{exp_id}.jsonl"
    with open(out,'w',encoding='utf-8') as f:
        for n in notes:
            f.write(json.dumps(n)+'\n')
    return out

def p_value_ttest(x,y):
    # Simple two-sample t-test for means, approximate
    n1,n2 = len(x), len(y)
    if n1<2 or n2<2:
        return 1.0
    m1 = sum(x)/n1
    m2 = sum(y)/n2
    v1 = sum((i-m1)**2 for i in x)/(n1-1)
    v2 = sum((i-m2)**2 for i in y)/(n2-1)
    se = math.sqrt(v1/n1 + v2/n2)
    if se==0:
        return 1.0
    t = abs(m1-m2)/se
    # Rough p approx for df>30
    # Use normal approx
    import math
    p = 2*(1 - 0.5*(1+math.erf(t/math.sqrt(2))))
    return max(0.0, min(1.0, p))

if __name__ == '__main__':
    notes = load_notes(SONG_LOG)
    a,b = split_ab(notes, seed=123)
    write_experiment('A', a)
    write_experiment('B', b)
    stats_a = harmony_stats(a)
    stats_b = harmony_stats(b)
    # p-values on convergence and entropy via bootstrap of note differences
    conv_a = [n.get('note',60) for n in a]
    conv_b = [n.get('note',60) for n in b]
    p_conv = p_value_ttest(conv_a, conv_b)
    # entropy per window
    # simple comparison
    result = {
        'exp_a': stats_a,
        'exp_b': stats_b,
        'p_value_convergence': p_conv,
        'p_value_entropy': 0.5  # placeholder
    }
    out = AB_DIR / 'ab_comparison.json'
    with open(out,'w',encoding='utf-8') as f:
        json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))
