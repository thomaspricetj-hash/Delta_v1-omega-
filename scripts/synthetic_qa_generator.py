"""
Synthetic QA generator for emergence acceleration
Creates paraphrased questions from existing knowledge
"""
import sys, json, random
from pathlib import Path
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "orchestrator"))
from master_orchestrator_v0 import teach_full

PREF_PATH = BASE / 'runtime' / 'logs' / 'synthetic_qa_state.json'

def load_state():
    if PREF_PATH.exists():
        return json.loads(PREF_PATH.read_text())
    return {"counter": 0}

def save_state(s):
    PREF_PATH.write_text(json.dumps(s))

# Sample fact templates
TEMPLATES = [
    ("What is the capital of {subject}?", "The capital of {subject} is {answer}."),
    ("Where is {subject} located?", "{subject} is located in {answer}."),
    ("What is {subject} known for?", "{subject} is known for {answer}."),
]

def generate_batch(n=100):
    # For demo, use a small set of seed facts
    seeds = [
        ("France", "Paris", "capital"),
        ("Germany", "Berlin", "capital"),
        ("Italy", "Rome", "capital"),
        ("Japan", "Tokyo", "capital"),
    ]
    for _ in range(n):
        subj, ans, _ = random.choice(seeds)
        q, a = random.choice(TEMPLATES)
        q = q.format(subject=subj, answer=ans)
        a = a.format(subject=subj, answer=ans)
        # Teach both forms
        teach_full(q, ans)
        teach_full(a, ans)

if __name__ == '__main__':
    state = load_state()
    generate_batch(200)
    state["counter"] += 200
    save_state(state)
    print(f"Synthetic QA generated: {state['counter']} total")
