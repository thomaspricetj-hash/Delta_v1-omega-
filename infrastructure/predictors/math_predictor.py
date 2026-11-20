"""
Math predictor for Translation OS v1
Simple arithmetic evaluation for queries
"""
import re
from infrastructure.predictors.predictor_wrapper import wrap_predictor

def _predict_impl(query):
    q = query or ""
    # Isolate last user turn to avoid history bleed
    if "User:" in q:
        q = q.rsplit("User:", 1)[-1]
    q_low = q.lower()
    # Simple addition pattern
    m = re.search(r'what is (\d+) plus (\d+)', q_low)
    if m:
        a = int(m.group(1)); b = int(m.group(2))
        return {'source':'math_predictor','response':str(a+b),'confidence':0.95}
    m = re.search(r'what is (\d+) times (\d+)', q_low)
    if m:
        a = int(m.group(1)); b = int(m.group(2))
        return {'source':'math_predictor','response':str(a*b),'confidence':0.95}
    return None

predict = wrap_predictor(_predict_impl, 'math_predictor')
