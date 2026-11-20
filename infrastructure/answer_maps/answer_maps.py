"""
Answer Maps with overlay convergence
Maintains separate multilayer foldable maps for correct and wrong answers
"""
import sys
from pathlib import Path
BASE = Path(__file__).resolve().parent.parent.parent

sys.path.insert(0, str(BASE / "infrastructure" / "foldable_map"))
from foldable_map import get_foldable_map

# Create two independent foldable maps
class AnswerMap:
    def __init__(self, name):
        self.name = name
        # Use separate instances by tagging with name in metadata
        self.fm = get_foldable_map()
    
    def put(self, query, answer, layer="answer", meta=None):
        meta = meta or {}
        meta.update({"map": self.name})
        self.fm.put(query, answer, layer=layer, meta=meta)
        # Index and ledger
        self._log_ledger(query, answer, layer, meta)
    
    def _log_ledger(self, query, answer, layer, meta):
        try:
            sys.path.insert(0, str(BASE / "infrastructure" / "master_map_integrator"))
            from master_map_integrator import log_ledger_entry
            log_ledger_entry(f"{self.name}_put", {"query":query, "answer":answer[:500], "layer":layer, "meta":meta})
        except Exception:
            pass

answer_map = AnswerMap("correct")
wrong_answer_map = AnswerMap("wrong")

def _token_overlap(a,b):
    sa=set(a.lower().split()); sb=set(b.lower().split())
    if not sa or not sb: return 0.0
    return len(sa & sb)/len(sa | sb)

def overlay_convergence(top_k=10):
    """
    Overlay correct and wrong maps to find convergence points
    Returns list of queries where both maps have entries with high similarity
    """
    ledger_path = BASE / "runtime" / "logs" / "master_ledger.jsonl"
    correct_entries=[]
    wrong_entries=[]
    try:
        from scripts.bitdrop_persistence import load_lines_bitdrop
        lines = load_lines_bitdrop(ledger_path)
        for e in lines:
            if e.get("type","").endswith("_put"):
                meta=e.get("payload",{}).get("meta",{})
                map_name=meta.get("map")
                q=e.get("payload",{}).get("query")
                ans=e.get("payload",{}).get("answer","")
                if not q: continue
                if map_name=="correct":
                    correct_entries.append((q,ans))
                elif map_name=="wrong":
                    wrong_entries.append((q,ans))
    except Exception:
        pass
    # Build similarity matrix
    results=[]
    for qc,ac in correct_entries:
        best_score=0.0
        best_qw=None
        best_aw=None
        for qw,aw in wrong_entries:
            qsim=_token_overlap(qc,qw)
            asym=_token_overlap(ac,aw)
            score=(qsim+asym)/2
            if score>best_score:
                best_score=score
                best_qw=qw
                best_aw=aw
        if best_score>0.3:
            results.append({"query_correct":qc,"query_wrong":best_qw,"convergence_score":round(best_score,3),"answer_correct":ac[:200],"answer_wrong":best_aw[:200]})
    # Sort and trigger reward push for top
    results.sort(key=lambda x:x["convergence_score"], reverse=True)
    top=results[:top_k]
    # Auto reward push: re-teach correct answer over wrong
    for r in top:
        try:
            sys.path.insert(0, str(BASE / "infrastructure" / "master_map_integrator"))
            from master_map_integrator import log_ledger_entry
            log_ledger_entry("convergence_reward_push", {"query_correct":r["query_correct"],"query_wrong":r["query_wrong"],"score":r["convergence_score"]})
            answer_map.fm.put(r["query_correct"], r["answer_correct"], layer="correct", meta={"map":"correct","reward_boost":True})
            # Prune low-weight wrong entries once convergence is high
            if r["convergence_score"] >= 0.7:
                log_ledger_entry("wrong_prune", {"query":r["query_wrong"],"reason":"high_convergence","score":r["convergence_score"]})
                # Mark for decay in foldable map
                wrong_answer_map.fm.put(r["query_wrong"], r["answer_wrong"], layer="wrong", meta={"map":"wrong","pruned":True,"decay":0.1})
        except Exception:
            pass
    return top

def log_wrong_answer(query, answer, reason="low_confidence"):
    wrong_answer_map.put(query, answer, layer="wrong", meta={"reason":reason})

def log_correct_answer(query, answer):
    answer_map.put(query, answer, layer="correct", meta={"source":"orchestrator"})

__all__ = ["answer_map","wrong_answer_map","log_correct_answer","log_wrong_answer","overlay_convergence"]
