"""
Rulebook predictor for Translation OS v1
Matches query against learned rule patterns
"""
import re
from infrastructure.predictors.predictor_wrapper import wrap_predictor
import json
from pathlib import Path

BASE = Path('C:/Users/thoma/.lmstudio/apps/bionic/projects/d49037d8-47f4-5808-9028-c707de117f8f/workspace/TRANSLATION_OS_V1')
RULEBOOK = BASE / 'runtime' / 'rulebook_patterns.json'

def _predict_impl():
    if not RULEBOOK.exists():
        return None
    try:
        patterns = json.loads(RULEBOOK.read_text())
    except Exception:
        return None
    q_low = query.lower()
    for pat, ans in patterns.items():
        if re.search(pat, q_low):
            return {'source':'rulebook_predictor','response':ans,'confidence':0.85}
    return None

predict = wrap_predictor(_predict_impl, "rulebook_predictor")
