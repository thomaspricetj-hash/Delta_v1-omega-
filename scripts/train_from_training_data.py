"""
Train Translation OS from external training data folder
Expects parquet files with SFT format: messages column with role/content
"""
import sys
from pathlib import Path
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "orchestrator"))
from master_orchestrator_v0 import teach_full

DATA_DIR = Path("../training data")
print("Scanning", DATA_DIR)

def teach_from_parquet(file_path):
    try:
        import pandas as pd
        df = pd.read_parquet(file_path)
    except Exception as e:
        print("Failed to read", file_path, e)
        return
    cols = [c.lower() for c in df.columns]
    pairs = []
    if "messages" in cols:
        for msgs in df["messages"]:
            q = a = None
            for m in msgs:
                role = m.get("role","").lower()
                content = m.get("content","")
                if role in ("user","human"):
                    q = content
                elif role in ("assistant","ai"):
                    a = content
            if q and a:
                pairs.append((q,a))
    elif "prompt" in cols and "completion" in cols:
        for _, row in df.iterrows():
            pairs.append((str(row["prompt"]), str(row["completion"])))
    elif "input" in cols and "output" in cols:
        for _, row in df.iterrows():
            pairs.append((str(row["input"]), str(row["output"])))
    elif "question" in cols and "answer" in cols:
        for _, row in df.iterrows():
            pairs.append((str(row["question"]), str(row["answer"])))
    else:
        print("Unknown schema in", file_path, df.columns.tolist())
        return
    print(f"Teaching {len(pairs)} pairs from {file_path.name}")
    for q,a in pairs:
        try:
            teach_full(q, a, reward=1.0)
        except Exception as e:
            print("teach error", e)
            continue

if __name__ == "__main__":
    if not DATA_DIR.exists():
        print("Data dir not found:", DATA_DIR)
        sys.exit(1)
    for p in sorted(DATA_DIR.glob("*.parquet")):
        teach_from_parquet(p)
    print("Done")
