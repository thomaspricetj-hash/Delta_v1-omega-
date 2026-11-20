"""
Reasoning trace persistence + retrieval
Stores full reasoning traces for Memory Reuse
"""
import json, time, sys
from pathlib import Path
BASE = Path(__file__).resolve().parent.parent
TRACE_PATH = BASE / "runtime" / "logs" / "reasoning_traces.jsonl"

def log_trace(query, diagnostics):
    entry = {
        "timestamp": time.time(),
        "query": query,
        "response": diagnostics.get("response"),
        "response_source": diagnostics.get("response_source"),
        "reasoning_thoughts": diagnostics.get("reasoning_thoughts"),
        "max3d_synthesis_response": diagnostics.get("max3d_synthesis_response"),
        "fiber_harmony_score": diagnostics.get("fiber_harmony_score"),
        "recursive_depth": diagnostics.get("recursive_depth"),
        "confidence_prior": diagnostics.get("confidence_prior")
    }
    # Use BitDrop v2 for persistence via bitdrop_persistence
    try:
        from bitdrop_persistence import save_line
        save_line(TRACE_PATH, entry)
    except Exception:
        with open(TRACE_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

def store_trace(query, answer, explanation, source="teacher", confidence=0.5):
    entry = {
        "timestamp": time.time(),
        "query": query,
        "answer": answer,
        "explanation": explanation,
        "source": source,
        "confidence": confidence
    }
    try:
        from bitdrop_persistence import save_line
        save_line(TRACE_PATH, entry)
    except Exception:
        with open(TRACE_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

def retrieve_similar(query, top_k=3):
    # Simple keyword overlap retrieval
    if not TRACE_PATH.exists():
        return []
    traces = []
    for line in TRACE_PATH.read_text().splitlines():
        try:
            t = json.loads(line)
            traces.append(t)
        except:
            pass
    q_words = set(query.lower().split())
    scored = []
    for t in traces:
        t_words = set(t.get("query","").lower().split())
        overlap = len(q_words & t_words)
        scored.append((overlap, t))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [s[1] for s in scored[:top_k]]

if __name__ == "__main__":
    print("Reasoning trace store ready")
