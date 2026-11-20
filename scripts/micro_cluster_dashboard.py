"""
Micro-cluster dashboard summary
Prints top 3 micro-clusters by size and emergence trend
"""
import json
from pathlib import Path
BASE = Path(__file__).resolve().parent.parent
from bitdrop_persistence import load_json_bitdrop

CLUSTER_PATH = BASE / "runtime" / "logs" / "query_clusters.json"
DASH_PATH = BASE / "runtime" / "logs" / "emergence_dashboard.jsonl"

def top_micro_clusters(n=3):
    data = load_json_bitdrop(CLUSTER_PATH)
    clusters = data.get("micro_clusters", []) if data else []
    # Sort by size
    sorted_cl = sorted(clusters, key=lambda c: len(c), reverse=True)
    top = sorted_cl[:n]
    return [(i, len(c), c[0] if c else None) for i, c in enumerate(top)]

def emergence_trend():
    # Simple trend: count of day_dream_priority logs
    if not DASH_PATH.with_suffix(DASH_PATH.suffix + ".bd2").exists():
        return "No data"
    # Load last 100 entries
    try:
        # BitDrop files are JSON lines
        lines = DASH_PATH.with_suffix(DASH_PATH.suffix + ".bd2").read_text(encoding="utf-8").splitlines()[-100:]
        count = 0
        for line in lines:
            try:
                entry = json.loads(line)
                if "day_dream_priority" in entry.get("metrics", {}):
                    count += 1
            except:
                pass
        return f"{count} priority logs in last 100 entries"
    except:
        return "No data"

if __name__ == "__main__":
    print("Top Micro Clusters:")
    for i, size, rep in top_micro_clusters():
        print(f"  #{i+1} size={size} representative={rep}")
    print("\nEmergence Trend:")
    print("  ", emergence_trend())
