"""
Active feedback reinforcement loop
"""
import json
import time
from pathlib import Path

BASE = Path('C:/Users/thoma/.lmstudio/apps/bionic/projects/d49037d8-47f4-5808-9028-c707de117f8f/workspace/TRANSLATION_OS_V1')
FEEDBACK_LOG = BASE / 'runtime' / 'logs' / 'feedback.jsonl'

def log_feedback(query, response, is_correct):
    entry = {
        'ts': time.time(),
        'query': query,
        'response': response,
        'is_correct': bool(is_correct),
        'reward': 1.0 if is_correct else -1.0
    }
    with open(FEEDBACK_LOG, 'a', encoding='utf-8') as f:
        f.write(json.dumps(entry)+'\n')
    # Update Fact Core confidence
    try:
        sys = __import__('sys')
        sys.path.insert(0, str(BASE / 'infrastructure' / 'fc_mbfi'))
        from fc_mbfi_engine import FactCore, FC_MBFIEngine
        parser = FC_MBFIEngine()
        subject, predicate = parser._parse_query(query)
        if subject and predicate and predicate != 'unknown':
            fc = FactCore()
            # Simple confidence update
            # In real system, adjust confidence based on reward
            pass
    except Exception:
        pass
    # Update Third Brain weights
    try:
        weights_path = BASE / 'runtime' / 'third_brain_weights.json'
        if weights_path.exists():
            w = json.loads(weights_path.read_text())
            # Shift weights slightly toward fact consistency on correct feedback
            if is_correct:
                w['fact_consistency'] = min(0.6, w.get('fact_consistency',0.35)+0.01)
            else:
                w['fact_consistency'] = max(0.2, w.get('fact_consistency',0.35)-0.01)
            # Normalize
            total = sum(w.values())
            for k in w:
                w[k] /= total
            weights_path.write_text(json.dumps(w))
    except Exception:
        pass

if __name__ == '__main__':
    import sys
    if len(sys.argv) >= 4:
        log_feedback(sys.argv[1], sys.argv[2], sys.argv[3].lower()=='true')
        print('Feedback logged')
