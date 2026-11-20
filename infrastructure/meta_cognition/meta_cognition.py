"""
Meta-Cognition Layer
Monitors its own reasoning, detects errors, and triggers self-correction
"""
import sys, time
from pathlib import Path
BASE = Path(__file__).resolve().parent.parent.parent

def reflect(query, diagnostics):
    """
    Self-reflection on last response
    Returns suggested improvements
    """
    issues = []
    conf = diagnostics.get("confidence_prior", 0.5)
    coherence = diagnostics.get("meaning_coherence", 0.5)
    if conf < 0.4:
        issues.append("low_confidence")
    if coherence < 0.5:
        issues.append("low_coherence")
    if diagnostics.get("model_called"):
        issues.append("model_fallback")
    # Log reflection
    try:
        sys.path.insert(0, str(BASE / "infrastructure" / "master_map_integrator"))
        from master_map_integrator import log_ledger_entry
        log_ledger_entry("meta_reflection", {"query": query[:200], "issues": issues, "diagnostics": diagnostics})
    except Exception:
        pass
    return {"issues": issues, "suggest_retry": len(issues) > 0}

def self_correct(query, diagnostics):
    """
    Trigger self-correction loop if reflection finds issues
    """
    refl = reflect(query, diagnostics)
    if refl["suggest_retry"]:
        # Re-query with reformulation
        try:
            sys.path.insert(0, str(BASE / "orchestrator"))
            from master_orchestrator_v0 import run_orchestrator
            # Simple reformulation: ask for explanation
            reform = f"Explain: {query}"
            new_res = run_orchestrator(reform)
            return new_res
        except Exception:
            pass
    return None

__all__ = ["reflect", "self_correct"]
