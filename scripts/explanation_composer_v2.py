"""
Explanation Composer V2 - template-based generative synthesis
Model-free prose generation from graph facts
"""
import sys
from pathlib import Path
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "orchestrator"))
from master_orchestrator_v0 import run_orchestrator

TEMPLATES = [
    " {topic} is a {type}. It is characterized by {props}. It relates to {rel}.",
    " {topic} is known for {props}. It is located in {rel}.",
    " {topic} refers to {type}. Common aspects include {props}.",
]

def synthesize(topic):
    # Retrieve facts directly via graph-aware explainer, avoid recursion
    try:
        from graph_aware_explainer import explain as graph_explain
        fact = graph_explain(topic)
    except:
        fact = ""
    # Simple template fill
    if not fact or "not in your knowledge graph" in fact.lower():
        return None
    # Heuristic parse
    type_ = "entity"
    props = "various properties"
    rel = "related concepts"
    # Very simple extraction
    if "." in fact:
        parts = fact.split(".")
        props = ". ".join(parts[:2])
    # Choose template
    tmpl = TEMPLATES[0]
    out = tmpl.format(topic=topic, type=type_, props=props, rel=rel).strip()
    return out

if __name__ == "__main__":
    print(synthesize("dog"))
