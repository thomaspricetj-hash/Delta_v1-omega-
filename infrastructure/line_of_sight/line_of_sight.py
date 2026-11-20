"""
Line of Sight module for Translation OS v1
Builds visibility graph from predictor graph and computes reachable neighborhood
"""
import json
from pathlib import Path

BASE = Path("TRANSLATION_OS_V1")
LOG_DIR = BASE / "runtime" / "logs"
GRAPH_PATH = LOG_DIR / "predictor_graph.json"

def load_graph():
    if not GRAPH_PATH.exists():
        return {"edges":[]}
    return json.loads(GRAPH_PATH.read_text())

def build_adjacency():
    g = load_graph()
    adj = {}
    for e in g.get("edges", []):
        src = e.get("from")
        dst = e.get("to")
        w = e.get("weight", 0.5)
        if not src or not dst:
            continue
        adj.setdefault(src, []).append((dst, w))
        adj.setdefault(dst, []).append((src, w))
    return adj

def line_of_sight(query, hops=2, threshold=0.5):
    adj = build_adjacency()
    # canonicalize query
    import re
    q = re.sub(r'[^\w\s]', ' ', query.lower())
    q = re.sub(r'\s+', ' ', q).strip()
    visited = set()
    frontier = [(q, 0)]
    visible = []
    while frontier:
        node, depth = frontier.pop(0)
        if node in visited or depth > hops:
            continue
        visited.add(node)
        for nb, w in adj.get(node, []):
            if w >= threshold:
                visible.append((nb, w, depth+1))
                if depth+1 < hops:
                    frontier.append((nb, depth+1))
    return visible

if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "What is the capital of France?"
    vis = line_of_sight(q, hops=2, threshold=0.5)
    print(json.dumps({"query":q,"visible":vis}, indent=2))
