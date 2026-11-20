"""
Master Overlay Predictor
Multilayered map with index and ledger overlays all predictions right/wrong for convergence
"""
import json
from infrastructure.predictors.predictor_wrapper import wrap_predictor
from pathlib import Path

BASE = Path('C:/Users/thoma/.lmstudio/apps/bionic/projects/d49037d8-47f4-5808-9028-c707de117f8f/workspace/TRANSLATION_OS_V1')
MASTER_MAP = BASE / 'runtime' / 'master_map_index.json'
LEDGER = BASE / 'runtime' / 'logs' / 'predictor_convergence.jsonl'

def _load_master():
    if MASTER_MAP.exists():
        try:
            return json.loads(MASTER_MAP.read_text())
        except Exception:
            return {}
    return {}

def _save_master(data):
    MASTER_MAP.write_text(json.dumps(data))

def _predict_impl():
    # Overlay all past predictions for this query
    qkey = query.lower().strip()
    master = _load_master()
    entry = master.get(qkey, {'hits':0,'correct':0,'wrong':0,'predictions':[]})
    # Simple convergence score
    hits = entry['hits']
    correct = entry['correct']
    total = max(1, hits)
    convergence = correct/total if hits>0 else 0.5
    # If convergence high, return best answer
    if hits >= 3 and convergence >= 0.8:
        best = max(entry['predictions'], key=lambda x:x.get('confidence',0), default=None)
        if best:
            return {'source':'master_overlay_predictor','response':best.get('answer'),'confidence':convergence,'convergence':convergence}
    return None

def log_prediction(query, answer, correct=None):
    # Log prediction for overlay
    qkey = query.lower().strip()
    master = _load_master()
    entry = master.get(qkey, {'hits':0,'correct':0,'wrong':0,'predictions':[]})
    entry['hits'] += 1
    entry['predictions'].append({'answer':answer,'ts':__import__('time').time()})
    if correct is not None:
        if correct:
            entry['correct'] += 1
        else:
            entry['wrong'] += 1
    master[qkey] = entry
    _save_master(master)

# Example hook: call log_prediction after each orchestrator response

predict = wrap_predictor(_predict_impl, "master_overlay_predictor")
