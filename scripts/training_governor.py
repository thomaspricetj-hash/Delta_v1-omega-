"""
Emergence dashboard → training governor
Auto-tunes 2h/day training mix based on emergence scores
"""
import json
from pathlib import Path
BASE = Path(__file__).resolve().parent.parent
DASH_PATH = BASE / "runtime" / "logs" / "emergence_dashboard.jsonl"
CONFIG_PATH = BASE / "runtime" / "logs" / "training_governor_config.json"

def load_recent_scores(n=50):
    scores = []
    if DASH_PATH.exists():
        lines = DASH_PATH.read_text().splitlines()[-n:]
        for line in lines:
            try:
                e = json.loads(line)
                scores.append(e)
            except:
                pass
    return scores

def compute_trend(scores):
    if not scores:
        return 0.0
    # Simple average of fiber_harmony_score if present
    vals = []
    for s in scores:
        m = s.get("metrics",{})
        if "fiber_harmony_score" in m:
            vals.append(m["fiber_harmony_score"])
    if not vals:
        return 0.0
    return sum(vals)/len(vals)

def adjust_params(trend):
    # If trend plateau <0.6, increase synthetic variation
    config = {
        "dream_cycles": 20,
        "synthetic_qa_batch": 200,
        "wikidata_batch_limit": 50000
    }
    if trend < 0.6:
        config["synthetic_qa_batch"] = 400
        config["dream_cycles"] = 30
    elif trend > 0.8:
        config["synthetic_qa_batch"] = 100
        config["dream_cycles"] = 10
    CONFIG_PATH.write_text(json.dumps(config, indent=2))
    return config

if __name__ == "__main__":
    scores = load_recent_scores()
    trend = compute_trend(scores)
    cfg = adjust_params(trend)
    print("Trend:", trend, "Config:", cfg)
