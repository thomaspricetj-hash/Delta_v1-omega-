"""
Temporal predictor — recency and decay weighting
"""
import json, time
from infrastructure.predictors.predictor_wrapper import wrap_predictor
from pathlib import Path

BASE = Path('C:/Users/thoma/.lmstudio/apps/bionic/projects/d49037d8-47f4-5808-9028-c707de117f8f/workspace/TRANSLATION_OS_V1')
FACT_CORE = BASE / 'runtime' / 'facts' / 'fact_core.jsonl'

def _predict_impl():
    if not FACT_CORE.exists():
        return None
    q_low = query.lower()
    best = None
    best_score = 0
    now = time.time()
    for line in FACT_CORE.read_text().splitlines():
        try:
            e = json.loads(line)
            # simple recency decay
            ts = e.get('ts', now)
            age_days = (now - ts) / 86400
            decay = max(0.1, 1.0 / (1 + age_days/30))
            if e.get('subject','').lower() in q_low:
                score = e.get('confidence',0.5) * decay
                if score > best_score:
                    best_score = score
                    best = e.get('object')
        except Exception:
            pass
    if best and best_score > 0.6:
        return {'source':'temporal_predictor','response':best,'confidence':best_score}
    return None

predict = wrap_predictor(_predict_impl, "temporal_predictor")
