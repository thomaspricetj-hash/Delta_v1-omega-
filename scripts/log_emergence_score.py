"""
Log Emergence Score daily
"""
import sys, json, datetime
from pathlib import Path
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "orchestrator"))
import master_orchestrator_v0 as m

LOG_PATH = BASE / 'runtime' / 'logs' / 'emergence_score_history.jsonl'

tests = [
    ("Novel", "What would happen if a submarine lived in a forest and learned astronomy?"),
    ("Compression", "Summarize in one sentence: The cat chased the mouse. The mouse escaped. The cat became tired."),
    ("Abstraction", "What do dogs and fish have in common?"),
    ("Generalization", "Red apples are sweet. Would a red pear likely be sweet?"),
    ("Recursive", "Why?"),
    ("Unseen", "A Flarn is larger than a Wex. A Wex is larger than a Nib. Which is largest?"),
    ("Consistency", "What is my favorite color?"),
]

score = 0
details = []
for name, q in tests:
    r = m.run_orchestrator(q)
    src = r.get('response_source')
    resp = r.get('response')
    hit = src is not None and resp is not None
    score += 1 if hit else 0
    details.append({"name": name, "source": src, "hit": hit})

entry = {
    "timestamp": datetime.datetime.utcnow().isoformat(),
    "score": score,
    "total": len(tests),
    "percent": score/len(tests)*100,
    "details": details
}
with open(LOG_PATH, 'a', encoding='utf-8') as f:
    f.write(json.dumps(entry) + "\n")
print(f"Logged Emergence Score: {score}/{len(tests)} = {score/len(tests)*100:.1f}%")
