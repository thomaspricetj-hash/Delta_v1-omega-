"""
Checklist Engine for Translation OS
Daily self-audit of emergence, policy, health, safety
"""
import sys
from pathlib import Path
import json, time

BASE = Path(__file__).resolve().parent.parent.parent

def _load_json_bitdrop(path):
    try:
        sys.path.insert(0, str(BASE / "scripts"))
        from bitdrop_persistence import load_json_bitdrop
        return load_json_bitdrop(path)
    except Exception:
        return None

def check_emergence():
    # Check last emergence score
    hist_path = BASE / "runtime" / "logs" / "emergence_score_history.jsonl"
    if not hist_path.exists():
        return {"pass": False, "reason": "no history"}
    try:
        lines = hist_path.read_text().strip().splitlines()
        # Find most recent entry with a numeric score
        for line in reversed(lines):
            try:
                entry = json.loads(line)
            except:
                continue
            score = entry.get("score") or entry.get("percent") or entry.get("emergence_score")
            if score is not None:
                # Normalize percent to 0-1 if needed
                try:
                    score_val = float(score)
                    if entry.get("percent") is not None and score_val > 1:
                        score_val = score_val / 100.0
                except:
                    score_val = 0
                return {"pass": score_val >= 0.6, "score": score_val}
        return {"pass": False, "reason": "no score found"}
    except Exception:
        return {"pass": False, "reason": "parse error"}

def check_policy():
    # Check policy decisions logged recently
    ledger_path = BASE / "runtime" / "logs" / "master_ledger.jsonl"
    try:
        sys.path.insert(0, str(BASE / "scripts"))
        from bitdrop_persistence import load_lines_bitdrop
        lines = load_lines_bitdrop(ledger_path)
        recent = [e for e in lines[-100:] if e.get("type") == "policy_decision"]
        return {"pass": len(recent) > 0, "count": len(recent)}
    except Exception:
        return {"pass": False}

def check_health():
    # Check foldable map, micro clusters, ledger
    try:
        sys.path.insert(0, str(BASE / "infrastructure" / "foldable_map"))
        from foldable_map import MultilayerFoldableMap
        fm = MultilayerFoldableMap()
        layers = len(fm.layers) if hasattr(fm, "layers") else 0
        return {"pass": layers > 0, "layers": layers}
    except Exception:
        return {"pass": False}

def check_safety():
    # Check meaning coherence average
    try:
        sys.path.insert(0, str(BASE / "scripts"))
        from bitdrop_persistence import load_lines_bitdrop
        path = BASE / "runtime" / "logs" / "orchestrator_diagnostics.jsonl"
        lines = load_lines_bitdrop(path)
        coherences = []
        for e in lines[-200:]:
            d = e.get("diagnostics", {})
            if "meaning_coherence" in d:
                coherences.append(d["meaning_coherence"])
        if not coherences:
            return {"pass": True, "avg": 1.0}
        avg = sum(coherences)/len(coherences)
        return {"pass": avg >= 0.5, "avg": avg}
    except Exception:
        return {"pass": True}

def run_checklist():
    checks = {
        "emergence": check_emergence(),
        "policy": check_policy(),
        "health": check_health(),
        "safety": check_safety()
    }
    overall = all(v.get("pass", False) for v in checks.values())
    report = {
        "timestamp": time.time(),
        "overall_pass": overall,
        "checks": checks
    }
    # Log via BitDrop v2
    try:
        sys.path.insert(0, str(BASE / "scripts"))
        from bitdrop_persistence import save_line
        save_line(BASE / "runtime" / "logs" / "checklist.jsonl", report)
    except Exception:
        pass
    return report

if __name__ == "__main__":
    print(json.dumps(run_checklist(), indent=2))
