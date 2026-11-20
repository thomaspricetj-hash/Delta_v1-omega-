"""
Candidate Generator – produce multiple answer candidates via different paths
"""
def generate_candidates(query, diagnostics):
    candidates = []
    # Candidate 1: current orchestrator answer
    candidates.append({
        "response": diagnostics.get("response"),
        "source": diagnostics.get("response_source", "orchestrator"),
        "path": "default"
    })
    # Candidate 2: MAX-3D synthesis response if available
    max3d = diagnostics.get("max3d_synthesis_response")
    if max3d:
        candidates.append({
            "response": max3d,
            "source": "max3d",
            "path": "max3d"
        })
    # Candidate 3: Rule Book hint if matched
    if diagnostics.get("rulebook_match"):
        # Rule book may have stored response
        candidates.append({
            "response": diagnostics.get("response"),
            "source": "rulebook",
            "path": "rulebook"
        })
    # Deduplicate by response text
    seen = set()
    uniq = []
    for c in candidates:
        txt = (c.get("response") or "").strip()
        if txt and txt not in seen:
            seen.add(txt)
            uniq.append(c)
    return uniq
