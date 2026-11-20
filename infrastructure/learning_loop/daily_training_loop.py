"""
Daily Training Loop for Translation OS V1
Runs a set of questions, checks answers, teaches design when wrong.
"""
import sys
from pathlib import Path
BASE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE / "runtime"))
sys.path.insert(0, str(BASE))

from runtime.translation_os_runtime import os_instance
from runtime.semantic_map import feedback, update_map
from runtime.predictive_cache import log_interaction
from runtime.pattern_mapper import update as pattern_update

TRAINING_SET = [
    ("What is the chemical symbol for gold?", "Au"),
    ("What is the speed of light in m/s?", "299792458"),
    ("Who discovered penicillin?", "Alexander Fleming"),
    ("What is the largest planet in our solar system?", "Jupiter"),
    ("What is the capital of Japan?", "Tokyo"),
    ("What is the time complexity of merge sort?", "O(n log n)"),
    ("What is the derivative of sin(x)?", "cos(x)"),
    ("What is the derivative of x squared?", "2x"),
    ("How many days in a week?", "7"),
    ("What is the capital of France?", "Paris"),
    ("Who wrote Romeo and Juliet?", "William Shakespeare"),
    ("What is 2+2?", "4"),
    ("Explain photosynthesis in one sentence", "photosynthesis converts sunlight"),
    ("What is machine learning in simple terms?", "machine learning allows computers to learn"),
    ("Why is the sky blue?", "scattering of sunlight"),
]

def check_answer(answer, key):
    return key.lower() in answer.lower()

def train():
    correct = 0
    taught = 0
    for q, key in TRAINING_SET:
        r = os_instance.chat(q)
        ans = r.get("answer","")
        src = r.get("source")
        ok = check_answer(ans, key)
        print(f"Q: {q}")
        print(f"  Answer: {ans[:100]} | src:{src} | ok:{ok}")
        if not ok:
            print("  -> teaching")
            feedback(q, ans, False)
            log_interaction(q, ans, {"source":"wrong"})
            good = f"{key} and is a learned response."
            update_map(q, good, weight=2.5)
            feedback(q, good, True)
            log_interaction(q, good, {"source":"correct"})
            pattern_update(q, good, reward=1.0)
            taught += 1
            r2 = os_instance.chat(q)
            print(f"  Re-check: {r2.get('answer')[:100]}")
        else:
            correct += 1
    print(f"Done. Correct:{correct} Taught:{taught} Total:{len(TRAINING_SET)}")

if __name__ == "__main__":
    train()
