"""
CLI for MBFN-augmented generation
"""
import argparse
from pathlib import Path
from infrastructure.mbfn.inference_wrapper import MBFNGenerator

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--hscl", required=False)
    p.add_argument("--prompt", required=True)
    p.add_argument("--steps", type=int, default=64)
    args = p.parse_args()
    gen = MBFNGenerator(Path(args.model), Path(args.hscl) if args.hscl else None)
    out = gen.generate(args.prompt, max_new_tokens=args.steps)
    print(out)

if __name__ == "__main__":
    main()
