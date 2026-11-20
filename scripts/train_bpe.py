"""
Train BPE tokenizer on training data
"""
import sys
import argparse
import random
from pathlib import Path
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
from infrastructure.tokenizer.bpe_tokenizer import get_bpe_tokenizer

def load_corpus(path):
    text_parts = []
    for p in Path(path).rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() in {".txt",".md",".jsonl",".json"}:
            try:
                txt = p.read_text(encoding="utf-8", errors="ignore")
                text_parts.append(txt)
            except Exception:
                pass
        elif p.suffix.lower() == ".parquet":
            try:
                import pandas as pd
                df = pd.read_parquet(p)
                # Try common text columns
                for col in ["text","content","prompt","completion","input","output"]:
                    if col in df.columns:
                        text_parts.append("\n".join(df[col].astype(str).tolist()))
                        break
                else:
                    # Fallback: join all string columns
                    for col in df.select_dtypes(include="object").columns:
                        text_parts.append("\n".join(df[col].astype(str).tolist()))
            except Exception:
                pass
    return "\n".join(text_parts)

def sample_corpus(corpus: str, max_chars: int, seed: int = 42) -> str:
    if len(corpus) <= max_chars:
        return corpus
    random.seed(seed)
    # Split into chunks of ~1MB for sampling
    chunk_size = 1_000_000
    chunks = [corpus[i:i+chunk_size] for i in range(0, len(corpus), chunk_size)]
    # Sample chunks proportionally to reach target size
    target_chunks = max(1, int(max_chars / chunk_size))
    sampled = random.sample(chunks, min(len(chunks), target_chunks * 2))
    # Build sampled corpus up to max_chars
    out = []
    total = 0
    for ch in sampled:
        if total >= max_chars:
            break
        take = min(len(ch), max_chars - total)
        out.append(ch[:take])
        total += take
    return "".join(out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BPE tokenizer")
    parser.add_argument("--max-chars", type=int, default=5_000_000, help="Maximum chars per chunk for training")
    parser.add_argument("--sample", action="store_true", help="Sample corpus instead of truncating head")
    parser.add_argument("--vocab-size", type=int, default=8192, help="Target vocab size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--chunks", type=int, default=1, help="Number of chunks to process incrementally, 1 = single pass")
    args = parser.parse_args()

    # Try trainingData inside TRANSLATION_OS_V1 first, then workspace/training data
    corpus_path = str(BASE / "trainingData")
    if not Path(corpus_path).exists():
        corpus_path = str(BASE.parent / "training data")
    print(f"Loading corpus from {corpus_path}")
    corpus = load_corpus(corpus_path)
    print(f"Original corpus size: {len(corpus)} chars")
    if len(corpus) > args.max_chars:
        if args.sample:
            print(f"Sampling corpus to {args.max_chars} chars with seed {args.seed}")
            corpus = sample_corpus(corpus, args.max_chars, args.seed)
        else:
            print(f"Corpus size exceeds MAX_CHARS {args.max_chars}, truncating head")
            corpus = corpus[:args.max_chars]
    print(f"Training corpus size: {len(corpus)} chars")
    tok = get_bpe_tokenizer()
    vocab_size = args.vocab_size
    print(f"Training BPE vocab_size={vocab_size}")
    tok.train(corpus, vocab_size=vocab_size, allowed_special={"<|endoftext|>"})
    out_dir = BASE / "runtime" / "tokenizer"
    out_dir.mkdir(parents=True, exist_ok=True)
    import json
    with open(out_dir / "vocab.json","w",encoding="utf-8") as f:
        json.dump(tok.vocab, f)
    # JSON can't serialize tuple keys
    merges_serializable = {f"{k[0]}:{k[1]}": v for k, v in tok.bpe_merges.items()}
    with open(out_dir / "bpe_merges.json","w",encoding="utf-8") as f:
        json.dump(merges_serializable, f)
    print(f"Saved vocab to {out_dir}")
