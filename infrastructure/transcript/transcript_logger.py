"""
Multilayered transcript with ledger index and map
"""
import json
import time
from pathlib import Path

BASE = Path('C:/Users/thoma/.lmstudio/apps/bionic/projects/d49037d8-47f4-5808-9028-c707de117f8f/workspace/TRANSLATION_OS_V1')
TRANSCRIPT_DIR = BASE / 'runtime' / 'transcripts'
LEDGER = TRANSCRIPT_DIR / 'ledger.jsonl'
INDEX = TRANSCRIPT_DIR / 'index.json'

TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)

def log_transcript(query, response, diagnostics):
    entry = {
        'ts': time.time(),
        'query': query,
        'response': response,
        'source': diagnostics.get('response_source'),
        'confidence': diagnostics.get('predictor_confidence'),
        'weave_attention': diagnostics.get('weave_attention'),
        'meta_eval': diagnostics.get('meta_evaluation'),
        'convergence': diagnostics.get('convergence')
    }
    # Append to ledger
    with open(LEDGER, 'a', encoding='utf-8') as f:
        f.write(json.dumps(entry)+'\n')
    # Update index map
    idx = {}
    if INDEX.exists():
        try:
            idx = json.loads(INDEX.read_text())
        except Exception:
            idx = {}
    qkey = query.lower().strip()
    if qkey not in idx:
        idx[qkey] = []
    idx[qkey].append({'ts': entry['ts'], 'response': response, 'source': entry['source']})
    INDEX.write_text(json.dumps(idx))
