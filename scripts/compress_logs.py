import json
from pathlib import Path
BASE = Path("C:/Users/thoma/.lmstudio/apps/bionic/projects/d49037d8-47f4-5808-9028-c707de117f8f/workspace/TRANSLATION_OS_V1")
import sys
sys.path.insert(0, str(BASE / "infrastructure"))
from bitdrop_v2_integration import compress_data

LOG_DIR = BASE / "runtime" / "logs"
out_dir = BASE / "runtime" / "logs_compressed"
out_dir.mkdir(exist_ok=True)

for p in LOG_DIR.glob("*.jsonl"):
    out_p = out_dir / (p.name + ".b2")
    total_orig = 0
    total_comp = 0
    with open(p, "rb") as f, open(out_p, "wb") as out:
        for line in f:
            if not line.strip():
                continue
            total_orig += len(line)
            # compress the line as bytes
            info = compress_data(line.rstrip(b"\n"))
            # store a tiny envelope
            envelope = {
                "meta": {
                    "original_len": info.get("original_len"),
                    "compressed_len": info.get("compressed_len"),
                    "ratio": info.get("ratio")
                },
                "line": line.decode("utf-8")
            }
            out.write(json.dumps(envelope).encode("utf-8") + b"\n")
            total_comp += len(json.dumps(envelope).encode("utf-8"))
    print(f"{p.name}: original {total_orig} -> envelope {total_comp} bytes, saved to {out_p.name}")
print("Done")
