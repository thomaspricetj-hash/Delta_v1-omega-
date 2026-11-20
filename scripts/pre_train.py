"""
Pre-training loop for Translation OS v1.
Trains BPE, rotary embeddings, and next-token LM with AdamW.
"""
import sys
from pathlib import Path
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "orchestrator"))

from infrastructure.tokenizer.bpe_tokenizer import train_bpe
from infrastructure.instruction_tuning.instruction_tuner import fine_tune
import json, time

LOGS = BASE / "runtime" / "logs"
LOGS.mkdir(parents=True, exist_ok=True)

def main():
    corpus_path = LOGS / "conversation.jsonl"
    if not corpus_path.exists():
        # fallback to self_supervised
        corpus_path = LOGS / "self_supervised.jsonl"
    print(f"Training BPE on {corpus_path}")
    train_bpe(corpus_path, vocab_size=8000)
    print("BPE complete")
    
    # Run a short pre-training warmup
    print("Running pre-training warmup...")
    from infrastructure.self_supervised.next_token_trainer import train_on_pair
    # Load some pairs
    pairs = []
    try:
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line)
                q = obj.get('query') or obj.get('prompt') or obj.get('user')
                a = obj.get('answer') or obj.get('assistant') or obj.get('response')
                if q and a:
                    pairs.append((q,a))
    except Exception:
        pairs = []
    
    if not pairs:
        print("No pairs found, exiting")
        return
    
    import random
    for i in range(500):
        q,a = random.choice(pairs)
        train_on_pair(q, a, reward=0.8)
        if i % 100 == 0:
            print(f"Pre-train step {i}")
    print("Pre-training warmup complete")
    print("Run instruction tuning next if you have instruction data.")

if __name__ == "__main__":
    main()
