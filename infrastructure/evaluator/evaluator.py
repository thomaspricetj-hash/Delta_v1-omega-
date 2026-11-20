"""
Evaluator – meta-cognitive layer that thinks about its thinking
Scores reasoning coherence, consistency, evidence coverage, and alignment.
"""
import json
from pathlib import Path
BASE = Path(__file__).resolve().parent.parent.parent
RUNTIME_LOGS = BASE / "runtime" / "logs"

def evaluate_reasoning(diagnostics, query):
    thoughts = diagnostics.get("reasoning_thoughts", [])
    fiber = diagnostics.get("fiber_harmony_score", 0.5)
    conf = diagnostics.get("confidence_prior", 0.5)
    north = diagnostics.get("north_star_alignment", {})
    alignment = north.get("alignment", 0.5) if isinstance(north, dict) else 0.5
    # Coherence: thoughts length vs complexity
    coherence = min(1.0, len(thoughts) / 3.0)
    # Consistency: fiber + alignment
    consistency = 0.6*fiber + 0.4*alignment
    # Evidence coverage: convergence hits
    conv = diagnostics.get("convergence_preload", {})
    evidence = min(1.0, conv.get("hits",0)/5.0 + 0.5)
    # Overall quality score
    quality = 0.3*coherence + 0.4*consistency + 0.3*evidence
    # Self-correction flag
    needs_rework = quality < 0.6 or conf < 0.4
    result = {
        "query": query,
        "coherence_score": round(coherence,3),
        "consistency_score": round(consistency,3),
        "evidence_coverage": round(evidence,3),
        "quality_score": round(quality,3),
        "needs_rework": needs_rework,
        "fiber_harmony_score": fiber,
        "confidence_prior": conf
    }
    # Log
    try:
        from scripts.bitdrop_persistence import save_line
        import sys
        sys.path.insert(0, str(BASE / "scripts"))
        from bitdrop_persistence import save_line
        save_line(RUNTIME_LOGS / "evaluator_logs.jsonl", result)
    except:
        pass
    return result

def meta_reflect(query, answer, diagnostics):
    eval_res = evaluate_reasoning(diagnostics, query)
    # Generate reflection note
    note = []
    if eval_res["needs_rework"]:
        note.append("Low quality – consider re-reasoning with alternative path")
    if eval_res["coherence_score"] < 0.5:
        note.append("Thin reasoning chain")
    if eval_res["consistency_score"] < 0.5:
        note.append("Fiber/North alignment weak")
    return {
        "evaluation": eval_res,
        "reflection": "; ".join(note) if note else "Reasoning looks solid"
    }
