"""
DTTQ CLI for production export/import
"""
import argparse
from pathlib import Path
from infrastructure.dttq.weight_tokenizer import export_compressed_model, import_compressed_model

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--vocab", type=int, default=4096)
    args = p.parse_args()
    export_compressed_model(Path(args.model), Path(args.out), vocab_size=args.vocab)
    print("Done")

if __name__ == "__main__":
    main()
