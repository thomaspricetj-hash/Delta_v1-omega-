"""
Math Pattern Micro-Helper
Boosts derivative/integral patterns and seeds semantic map for math queries.
"""
import re
from pathlib import Path
import sys
BASE = Path(__file__).resolve().parent.parent.parent

def detect_math_pattern(query):
    q = query.lower()
    if "derivative of" in q:
        return "derivative"
    if "integral of" in q:
        return "integral"
    if "what is" in q and any(op in q for op in ["+", "-", "*", "/", "^", "%"]):
        return "arithmetic"
    return None

def boost_semantic_candidates(query, candidates):
    """
    Boost scores for math patterns based on operator keywords.
    """
    pattern = detect_math_pattern(query)
    if not pattern:
        return candidates
    boosted = []
    for c in candidates:
        score = c.get("score",0)
        # Boost if candidate contains math-like tokens
        ans = c.get("answer","").lower()
        if pattern == "derivative":
            if any(tok in ans for tok in ["x ", "dx", "^", "squared", "cubed", "e^"]):
                score += 10.0
        if pattern == "integral":
            if any(tok in ans for tok in ["x ", "dx", "squared", "cubed", "sin", "cos"]):
                score += 10.0
        c2 = dict(c)
        c2["score"] = score
        boosted.append(c2)
    return boosted

def ensure_math_meanings():
    # Seed word meaning mapper with math operators
    try:
        sys.path.insert(0, str(BASE / "runtime"))
        from word_meaning_mapper import add_meaning
        operators = {
            "derivative": "rate of change",
            "integral": "accumulation of area",
            "dx": "infinitesimal change",
            "x^": "power of x",
            "e^x": "exponential function"
        }
        for w, m in operators.items():
            add_meaning(w, m, weight=1.0)
    except Exception:
        pass

if __name__ == "__main__":
    ensure_math_meanings()
    print("Math pattern helper initialized")
