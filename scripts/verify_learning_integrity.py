"""
Verify learning integrity logs for full-design passes
"""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
LOG_PATH = BASE / "runtime" / "logs" / "learning_integrity.jsonl"

EXPECTED_HOOKS = [
    "hscl","abbrev","master_map","semantic_map","context_cache","token_predictor",
    "snapshot","hybrid_cognitive","mdwe","kv_webb","weave","future_book",
    "router","integrator","structured","self_supervised","tiny_transformer",
    "real_transformer","multihead","self_study","red_team"
]

def check_entry(entry):
    hooks = entry.get("hooks", [])
    missing = [h for h in EXPECTED_HOOKS if h not in hooks]
    ok = entry.get("full_design_pass") is True and not missing
    return ok, missing

def main(n=20):
    if not LOG_PATH.exists():
        print(f"No integrity log found at {LOG_PATH}")
        return
    lines = LOG_PATH.read_text().splitlines()
    total = len(lines)
    print(f"Integrity log entries: {total}")
    print("-"*80)
    for line in lines[-n:]:
        try:
            e = json.loads(line)
        except:
            continue
        ok, missing = check_entry(e)
        status = "PASS" if ok else "FAIL"
        q = e.get("query","")[:80]
        print(f"[{status}] {e.get('ts')} | {q} ... | hooks={len(e.get('hooks',[]))}")
        if missing:
            print(f"  Missing hooks: {missing}")
    print("-"*80)
    # Summary
    passes = 0
    fails = 0
    for line in lines:
        try:
            e = json.loads(line)
            ok,_ = check_entry(e)
            if ok: passes+=1
            else: fails+=1
        except:
            continue
    print(f"Summary: {passes} PASS, {fails} FAIL out of {passes+fails} entries")

if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv)>1 else 20
    main(n)
