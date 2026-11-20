"""
Generate synthetic math QA using matrix_multiply_predictor
Feeds cheap math into model training so it learns to predict its own math
"""
import random
import json
from pathlib import Path
import sys
BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from infrastructure.predictors.matrix_multiply_predictor import cached_matmul

OPS = ['*','+','-','/']
def gen_pair():
    a = random.randint(0, 1000)
    b = random.randint(1, 1000)
    op = random.choice(OPS)
    if op == '/' and b == 0:
        b = 1
    q = f"What is {a} {op} {b}?"
    res = cached_matmul(a, b, op)
    if res is None:
        return None
    ans = str(int(res)) if float(res).is_integer() else str(res)
    return {"question": q, "answer": ans, "op": op, "a": a, "b": b}

def main(n=10000, out="runtime/logs/math_synthetic.jsonl"):
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for _ in range(n):
            pair = gen_pair()
            if pair:
                f.write(json.dumps(pair) + "\n")
    print(f"Wrote {n} synthetic math pairs to {out_path}")

if __name__ == "__main__":
    main()
