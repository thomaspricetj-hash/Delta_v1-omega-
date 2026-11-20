"""
Multilayered Weave with Lattice Interconnect and Interference
Weaves semantic, pattern, next-word, statistical maps into a lattice.
Interference penalises wrong word/answer pairs to rule out incorrect mappings.
"""
import json
from pathlib import Path
import re
from collections import defaultdict

BASE = Path(__file__).resolve().parent.parent.parent
RUNTIME_LOGS = BASE / "runtime" / "logs"
WEAVE_STATE = RUNTIME_LOGS / "lattice_weave_state.json"

def _get_bitdrop_paths():
    import sys
    sys.path.insert(0, str(BASE / "scripts"))
    from bitdrop_persistence import save_json, load_json_bitdrop
    return save_json, load_json_bitdrop

def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

def load_state():
    try:
        save_json, load_json_bitdrop = _get_bitdrop_paths()
        data = load_json_bitdrop(WEAVE_STATE)
        if data:
            return data
    except Exception:
        pass
    if WEAVE_STATE.exists():
        try:
            return json.loads(WEAVE_STATE.read_text(encoding="utf-8"))
        except:
            pass
    return {"lattice": {}, "interference": {}, "history": []}

def save_state(state):
    try:
        save_json, _ = _get_bitdrop_paths()
        save_json(WEAVE_STATE, state)
    except Exception:
        try:
            WEAVE_STATE.parent.mkdir(parents=True, exist_ok=True)
            with open(WEAVE_STATE, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
        except:
            pass

def weave_layers(query, answer, weight=1.0):
    state = load_state()
    lattice = state.setdefault("lattice", {})
    interference = state.setdefault("interference", {})
    words = tokenize(query)
    # Connect query words to answer in lattice
    for w in words:
        node_key = f"w:{w}"
        ans_key = f"a:{answer}"
        lattice.setdefault(node_key, {"type":"word","links":[]})
        lattice.setdefault(ans_key, {"type":"answer","links":[]})
        # Add bidirectional links with weight
        if ans_key not in lattice[node_key]["links"]:
            lattice[node_key]["links"].append(ans_key)
        if node_key not in lattice[ans_key]["links"]:
            lattice[ans_key]["links"].append(node_key)
        # Update interference scores: correct pairs get positive weight
        pair = f"{w}||{answer}"
        interference[pair] = interference.get(pair, 0.0) + weight
    state["history"].append({"query":query,"answer":answer,"weight":weight})
    save_state(state)
    return {"woven": True, "words": words, "answer": answer}

def apply_interference(query, candidates):
    """
    candidates: list of dicts with 'answer' and 'score'
    Returns filtered candidates with interference penalty for wrong word/answer pairs.
    """
    state = load_state()
    interference = state.get("interference", {})
    words = tokenize(query)
    filtered = []
    for cand in candidates:
        ans = cand["answer"]
        # Compute interference penalty
        penalty = 0.0
        for w in words:
            pair = f"{w}||{ans}"
            # If this pair has been marked as wrong before, penalty applies
            # We track negative interference via separate wrong_log in semantic map, here we approximate
            # For now, penalize if no positive interference exists
            if interference.get(pair,0) <= 0:
                penalty += 0.5
        score = cand.get("score",0) - penalty
        if score > 0:
            filtered.append({**cand, "score": score, "interference_penalty": penalty})
    # Sort by adjusted score
    filtered.sort(key=lambda x: x["score"], reverse=True)
    return filtered

def get_lattice_summary():
    state = load_state()
    lattice = state.get("lattice", {})
    # Simple summary: count nodes and links
    nodes = len(lattice)
    links = sum(len(v.get("links",[])) for v in lattice.values()) // 2
    return {"nodes": nodes, "links": links, "history_len": len(state.get("history",[]))}

if __name__ == "__main__":
    print(json.dumps(get_lattice_summary(), indent=2))
