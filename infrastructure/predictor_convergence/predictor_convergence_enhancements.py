"""
Predictor Convergence Enhancements
"""
import json, time, math
from pathlib import Path
BASE = Path("TRANSLATION_OS_V1")
LOG_DIR = BASE / "runtime" / "logs"
LEDGER = LOG_DIR / "predictor_ledger.jsonl"
CONV = LOG_DIR / "predictor_convergence.jsonl"
MAP = LOG_DIR / "predictor_map.json"
GRAPH = LOG_DIR / "predictor_graph.json"
SUMMARY = LOG_DIR / "predictor_convergence_summary.json"

def reward_weighted_stability(confidences, rewards):
    if not confidences: 
        return 1.0
    w = [max(0.1, abs(r)) for r in rewards]
    total = sum(w)
    mean = sum(c*w_i for c,w_i in zip(confidences,w))/total
    var = sum(w_i*(c-mean)**2 for c,w_i in zip(confidences,w))/total
    return max(0.0, 1.0 - math.sqrt(var))

def load_past(qcanon, predictor):
    past_conf, past_rewards = [], []
    if LEDGER.exists():
        with open(LEDGER) as f:
            for line in f:
                e = json.loads(line)
                if e.get("qcanon")==qcanon and e.get("predictor")==predictor:
                    past_conf.append(e.get("confidence",0))
                    past_rewards.append(e.get("reward",1.0))
    return past_conf, past_rewards

def adaptive_horizon(conf):
    h = int(conf*7)
    return max(1, min(7, h))

def update_dashboard():
    summary = {}
    if CONV.exists():
        agg = {}
        with open(CONV) as f:
            for line in f:
                e = json.loads(line)
                q = e.get("qcanon")
                if q not in agg:
                    agg[q] = {"scores":[], "stability":[], "overlay":[]}
                agg[q]["scores"].append(e.get("convergence_score",0))
                agg[q]["stability"].append(e.get("past",{}).get("stability",1))
                agg[q]["overlay"].append(e.get("overlay"))
        for q,v in agg.items():
            scores = v["scores"]
            summary[q] = {
                "last_score": scores[-1],
                "mean_score": sum(scores)/len(scores),
                "stability_mean": sum(v["stability"])/len(v["stability"]),
                "overlay_trend": v["overlay"][-5:]
            }
    with open(SUMMARY,"w") as f:
        json.dump(summary,f)

if __name__ == "__main__":
    update_dashboard()
