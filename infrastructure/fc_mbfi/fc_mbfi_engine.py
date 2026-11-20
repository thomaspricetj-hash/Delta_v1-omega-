"""
Fact-Centric Multi-Layer Binary Frame Intelligence Architecture integration
"""
import json
import time
from pathlib import Path
from collections import defaultdict

BASE = Path('C:/Users/thoma/.lmstudio/apps/bionic/projects/d49037d8-47f4-5808-9028-c707de117f8f/workspace/TRANSLATION_OS_V1')
FACT_DIR = BASE / 'runtime' / 'facts'
FACT_DIR.mkdir(parents=True, exist_ok=True)

FACT_CORE_PATH = FACT_DIR / 'fact_core.jsonl'
FRAME_STORE_PATH = FACT_DIR / 'frames.jsonl'

class FactCore:
    def __init__(self):
        self.facts = {}
        self._load()
    def _load(self):
        if FACT_CORE_PATH.exists():
            for line in FACT_CORE_PATH.open('r', encoding='utf-8'):
                try:
                    e = json.loads(line)
                    k = (e['subject'], e['predicate'], e['object'])
                    self.facts[k] = e
                except Exception:
                    pass
    def add_fact(self, subject, predicate, obj, confidence=1.0, source='teach'):
        import hashlib, json
        key = (subject,predicate,obj)
        # Idempotent with delta tracking
        existing = self.facts.get(key)
        entry = {'subject':subject,'predicate':predicate,'object':obj,'confidence':confidence,'source':source,'ts':time.time()}
        if existing:
            # Delta only: update confidence and source, log delta
            delta = {
                'subject':subject,'predicate':predicate,'object':obj,
                'prev_confidence':existing.get('confidence'),
                'new_confidence':confidence,
                'delta_confidence': round(confidence - existing.get('confidence',0),4),
                'repeat_count': existing.get('repeat_count',0)+1,
                'last_seen': entry['ts'],
                'source':source
            }
            existing['confidence'] = confidence
            existing['source'] = source
            existing['repeat_count'] = delta['repeat_count']
            # Append delta to cold delta log
            delta_path = FACT_DIR / 'fact_deltas.jsonl'
            with delta_path.open('a', encoding='utf-8') as df:
                df.write(json.dumps(delta)+"\n")
            self.facts[key] = existing
            return
        # New fact
        entry['repeat_count'] = 1
        self.facts[key] = entry
        with FACT_CORE_PATH.open('a', encoding='utf-8') as f:
            f.write(json.dumps(entry)+'\n')
    def lookup(self, subject, predicate):
        results = []
        for (s,p,o),e in self.facts.items():
            if s==subject and p==predicate:
                results.append(e)
        return results
    def verify(self, subject, predicate, obj):
        key = (subject,predicate,obj)
        e = self.facts.get(key)
        if e:
            return True, e.get('confidence',1.0)
        return False, 0.0

class BinaryFrameNetwork:
    def __init__(self):
        self.frames = {}
        self._load()
    def _load(self):
        if FRAME_STORE_PATH.exists():
            for line in FRAME_STORE_PATH.open('r', encoding='utf-8'):
                try:
                    e = json.loads(line)
                    self.frames[e['id']] = e
                except Exception:
                    pass
    def activate(self, concept):
        # Simple activation: return frames where concept is related
        activated = []
        for fid, f in self.frames.items():
            if concept.lower() in f.get('meaning','').lower():
                activated.append({'id':fid,'state':'support','strength':f.get('strength',0.5)})
        return activated
    def vote(self, candidates):
        # candidates = list of {frame_id, support, oppose}
        scores = defaultdict(float)
        for c in candidates:
            fid = c.get('frame_id')
            support = c.get('support',0)
            oppose = c.get('oppose',0)
            scores[fid] += support - oppose
        best = max(scores.items(), key=lambda x:x[1]) if scores else (None,0)
        return best

class FC_MBFIEngine:
    def __init__(self):
        self.fact_core = FactCore()
        self.frame_net = BinaryFrameNetwork()
    def _parse_query(self, query):
        q = query.lower()
        # Patterns
        import re
        patterns = [
            (r'what is (?:the )?capital of ([a-zA-Z ]+)\??', 'capital', lambda m: m.group(1).strip().title()),
            (r'capital of ([a-zA-Z ]+)\??', 'capital', lambda m: m.group(1).strip().title()),
            (r'what is ([a-zA-Z ]+) capital\??', 'capital', lambda m: m.group(1).strip().title()),
            (r'what is the speed of light', 'speed_of_light', lambda m: ''),
            (r'what is 42 plus 42\?', '42_plus_42', lambda m: ''),
            (r'what is ([\d ]+) plus ([\d ]+)\?', 'plus', lambda m: f"{m.group(1).strip()} {m.group(2).strip()}"),
        ]
        for pat, pred, subj_fn in patterns:
            m = re.search(pat, q)
            if m:
                subject = subj_fn(m)
                return subject, pred
        # fallback heuristic
        words = re.findall(r'[A-Za-z]{4,}', q)
        if words:
            subject = words[-1].title()
            predicate = 'unknown'
            return subject, predicate
        return None, None
    def process(self, query):
        subject, predicate = self._parse_query(query)
        trace = {'query': query, 'parsed_subject': subject, 'parsed_predicate': predicate}
        if not subject:
            return {'type':'PREDICTION','confidence':0.1,'answer':'Unable to parse query','trace':trace}
        # Fact verification attempts
        # Try common object candidates from fact core
        candidates = self.fact_core.lookup(subject, predicate)
        if candidates:
            # pick highest confidence
            best = max(candidates, key=lambda e: e.get('confidence',0.0))
            trace['fact_core_hit']=True
            trace['fact']=best
            return {'type':'KNOWN_FACT','confidence':best.get('confidence',1.0),'answer':best.get('object'),'trace':trace,'confidence_separation':'KNOWN FACT'}
        # Try generic lookup: subject predicate ?
        # Attempt to find any fact with subject and predicate
        # Frame activation for reasoning
        frames = self.frame_net.activate(subject)
        trace['frames']=frames
        # Voting over candidates from frames
        # If no fact, do prediction with confidence 0.5
        return {'type':'PREDICTION','confidence':0.5,'answer':'Unknown','trace':trace,'confidence_separation':'PREDICTION'}
