"""
Instruction tuning loader and fine-tune loop.
Loads instruction datasets and runs supervised fine-tuning on the native LM.
"""
from pathlib import Path
import json

BASE = Path(__file__).resolve().parents[2]
DATA_DIR = BASE / "training data"
LOGS = BASE / "runtime" / "logs"
LOGS.mkdir(parents=True, exist_ok=True)

def load_instructions():
    pairs = []
    # Load any jsonl files in training data
    for p in DATA_DIR.glob("*.jsonl"):
        try:
            for line in p.open(encoding='utf-8'):
                obj = json.loads(line)
                q = obj.get('instruction') or obj.get('query') or obj.get('prompt')
                a = obj.get('output') or obj.get('answer') or obj.get('response')
                if q and a:
                    pairs.append((q, a))
        except Exception:
            pass
    return pairs

def fine_tune(steps=1000, lr=3e-4):
    from infrastructure.self_supervised.next_token_trainer import train_on_pair
    pairs = load_instructions()
    if not pairs:
        print("No instruction data found")
        return
    import random
    for i in range(steps):
        q,a = random.choice(pairs)
        train_on_pair(q, a, reward=1.0)
        if i % 100 == 0:
            print(f"Instruction fine-tune step {i}/{steps}")
    print("Instruction tuning complete")
