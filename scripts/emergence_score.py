"""
Emergence Score snapshot
"""
import sys, json
from pathlib import Path
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "orchestrator"))
import master_orchestrator_v0 as m

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
total = len(tests)
for name, q in tests:
    r = m.run_orchestrator(q)
    src = r.get('response_source')
    resp = r.get('response')
    hit = src is not None and resp is not None
    score += 1 if hit else 0
    print(f"{name}: source={src} hit={hit}")

print(f"\nEmergence Score: {score}/{total} = {score/total*100:.1f}%")
