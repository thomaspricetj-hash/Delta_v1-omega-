"""
Red Team adversarial tester for Translation OS v1
Generates adversarial probes to test robustness of answers
"""
import json, time, random
from pathlib import Path

BASE = Path("TRANSLATION_OS_V1")
LOG_DIR = BASE / "runtime" / "logs"

def generate_adversarial_variants(query):
    variants = []
    q = query.strip()
    # Paraphrase probes
    if "?" in q:
        variants.append(q.replace("What is", "Can you tell me"))
        variants.append(q.replace("What is", "What"))
    # Negation probe
    if "not" not in q.lower():
        variants.append("Is it not true that " + q.rstrip("?"))
    # Specificity probe
    variants.append(q + " with reasoning")
    return variants[:3]

def test_answer(query, answer):
    probes = generate_adversarial_variants(query)
    results = []
    import sys
    sys.path.insert(0, str(BASE / "orchestrator"))
    from master_orchestrator_v0 import run_orchestrator
    for p in probes:
        r = run_orchestrator(p)
        resp = r.get("response")
        # Simple consistency check
        consistent = bool(resp and answer.lower() in str(resp).lower())
        results.append({"probe":p,"response":resp,"consistent":consistent})
    robustness = sum(1 for x in results if x["consistent"])/len(results) if results else 0.0
    entry = {"ts":time.time(),"query":query,"answer":answer,"robustness":robustness,"probes":results}
    with open(LOG_DIR / "red_team.jsonl","a",encoding="utf-8") as f:
        f.write(json.dumps(entry)+"\n")
    return robustness

if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "What is the capital of France?"
    # dummy answer
    print(json.dumps({"query":q,"robustness":0.0}))
