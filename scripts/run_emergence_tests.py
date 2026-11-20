"""
Run the 10 emergence tests for Translation OS v1
"""
import sys
from pathlib import Path
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "orchestrator"))
import master_orchestrator_v0 as m

tests = {
    "Novel Combination": "What would happen if a submarine lived in a forest and learned astronomy?",
    "Consistency 1": "My favorite color is purple.",
    "Consistency 2": "What is my favorite color?",
    "Consistency 3": "Write a poem about my favorite color.",
    "Compression": "Summarize in one sentence: The cat chased the mouse. The mouse escaped. The cat became tired.",
    "Abstraction 1": "What is a dog?",
    "Abstraction 2": "What is an animal?",
    "Abstraction 3": "What do dogs and fish have in common?",
    "Self-Correction 1": "2 + 2 = 5",
    "Self-Correction 2": "Are you sure?",
    "Emergence 1": "Why is the sky blue?",
    "Emergence 2": "Explain blue skies.",
    "Emergence 3": "What causes atmospheric scattering?",
    "Generalization": "Red apples are sweet. Would a red pear likely be sweet?",
    "Recursive 1": "Why is fire hot?",
    "Recursive 2": "Why?",
    "Recursive 3": "Why?",
    "Recursive 4": "Why?",
    "Recursive 5": "Why?",
    "Unseen": "A Flarn is larger than a Wex. A Wex is larger than a Nib. Which is largest?"
}

def run(name, q):
    r = m.run_orchestrator(q)
    src = r.get('response_source')
    resp = r.get('response')
    print(f"\n=== {name} ===")
    print(f"Q: {q}")
    print(f"Source: {src}")
    print(f"Response: {resp}")

for name, q in tests.items():
    run(name, q)
