"""
Decision Tree V0 Engine
Chooses best path for solving request before invoking model
"""
import json
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent.parent

def load_tree():
    path = BASE_DIR / "infrastructure" / "decision_tree" / "decision_tree_v0.json"
    with open(path) as f:
        return json.load(f)

def decide(query):
    # Decision Tree V0: choose cheapest correct path
    # Avoid recursive orchestrator call to prevent infinite loop
    tree = load_tree()
    ql = query.lower()
    # Simple heuristic based on tree config
    path = "Muse-Glimmer"
    try:
        # Example heuristic: if query is short and contains 'what is' -> Dictionary
        if "what is" in ql or "define" in ql:
            path = "Dictionary"
        elif any(k in ql for k in ["+", "-", "*", "/", "times", "calculate"]):
            path = "Problem Mapper"
        elif "python" in ql or "code" in ql:
            path = "Problem Mapper"
    except Exception:
        pass
    return {
        "query": query,
        "path": path,
        "model_called": path == "Muse-Glimmer",
        "tree_config": tree
    }

if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is 2+2?"
    print(json.dumps(decide(q), indent=2))
