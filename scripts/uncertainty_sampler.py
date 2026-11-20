"""
Uncertainty-driven active learning sampler
Prioritizes day_dream_queue items with low confidence or mid fiber score
"""
import json
from pathlib import Path
BASE = Path(__file__).resolve().parent.parent
QUEUE_PATH = BASE / "runtime" / "logs" / "day_dream_queue.jsonl"
CONF_PATH = BASE / "runtime" / "logs" / "confidence_state.json"

def load_conf():
    try:
        return json.loads(CONF_PATH.read_text())
    except:
        return {}

def sample_uncertain(n=10):
    conf = load_conf()
    items = []
    if QUEUE_PATH.exists():
        for line in QUEUE_PATH.read_text().splitlines():
            try:
                e = json.loads(line)
                q = e.get("query","").lower()
                c = conf.get(q,{}).get("confidence",0.5)
                # fiber score placeholder
                score = 0.5
                # uncertainty = distance from 0.5
                uncertainty = abs(0.5 - c)
                items.append((uncertainty, c, e))
            except:
                pass
    # Sort by uncertainty ascending? We want low confidence
    items.sort(key=lambda x: x[1])
    return [i[2] for i in items[:n]]

if __name__ == "__main__":
    s = sample_uncertain(5)
    print(json.dumps(s, indent=2))
