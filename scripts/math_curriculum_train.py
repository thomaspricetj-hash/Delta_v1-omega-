"""
Reverse curriculum training for matrix multiply predictor
Start with correct answer first, work backwards to harder operands
"""
import random
import json
from pathlib import Path
import sys
BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from infrastructure.predictors.matrix_multiply_predictor import cached_matmul

OPS = ['*','+','-','/']

def gen_easy():
    a = random.randint(0, 20)
    b = random.randint(1, 20)
    op = random.choice(OPS)
    if op == '/' and b == 0:
        b = 1
    return a, b, op

def gen_hard():
    a = random.randint(0, 1000)
    b = random.randint(1, 1000)
    op = random.choice(OPS)
    return a, b, op

def curriculum_batch(size, stage):
    # stage 0-1 easy, 1-2 medium, >2 hard
    if stage < 1:
        gen = gen_easy
    elif stage < 2:
        # medium
        a = random.randint(0, 100)
        b = random.randint(1, 100)
        op = random.choice(OPS)
        return [(a,b,op) for _ in range(size)]
    else:
        gen = gen_hard
    return [gen() for _ in range(size)]

def main(steps=1000, batch_size=32):
    out_path = Path('runtime/logs/math_curriculum.jsonl')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        for step in range(steps):
            stage = min(3, step / 300)  # progress difficulty
            batch = curriculum_batch(batch_size, stage)
            for a,b,op in batch:
                res = cached_matmul(a,b,op)
                if res is None:
                    continue
                # Forward example: expression -> answer
                f.write(json.dumps({
                    "type": "forward",
                    "question": f"What is {a} {op} {b}?",
                    "answer": str(res),
                    "a": a, "b": b, "op": op, "stage": stage
                }) + "\n")
                # Reverse example: answer -> expression, start with correct answer first
                # This is the "work backwards" curriculum
                f.write(json.dumps({
                    "type": "reverse",
                    "prompt": f"The answer is {res}. What operation and operands produce it?",
                    "answer": f"{a} {op} {b}",
                    "a": a, "b": b, "op": op, "stage": stage
                }) + "\n")
            if step % 100 == 0:
                print(f"Step {step} stage {stage:.2f}")
    print(f"Wrote curriculum to {out_path}")

if __name__ == "__main__":
    main()
