"""
Third Brain – autonomous learning loop inside orchestrator
Pulls own data from storage, self-grades with weighted feedback, info braiding, connection making, support vectors with deltas.
"""
import json
import time
from pathlib import Path
from collections import defaultdict
import random

BASE = Path('C:/Users/thoma/.lmstudio/apps/bionic/projects/d49037d8-47f4-5808-9028-c707de117f8f/workspace/TRANSLATION_OS_V1')
RUNTIME = BASE / 'runtime'
LOGS = RUNTIME / 'logs'
FACT_DIR = RUNTIME / 'facts'
STORAGE_PATHS = [
    LOGS / 'predictor_convergence.jsonl',
    LOGS / 'orchestrator_diagnostics.jsonl',
    FACT_DIR / 'fact_core.jsonl',
    RUNTIME / 'embeddings' / 'embed_store_index.json',
]

class ThirdBrain:
    def __init__(self):
        self.weights_path = RUNTIME / 'third_brain_weights.json'
        if self.weights_path.exists():
            try:
                self.weights = json.loads(self.weights_path.read_text())
            except Exception:
                self.weights = {
                    'fact_consistency': 0.35,
                    'predictor_confidence': 0.25,
                    'convergence_stability': 0.20,
                    'novelty': 0.10,
                    'feedback_signal': 0.10
                }
        else:
            self.weights = {
                'fact_consistency': 0.35,
                'predictor_confidence': 0.25,
                'convergence_stability': 0.20,
                'novelty': 0.10,
                'feedback_signal': 0.10
            }
        self.support_vectors = defaultdict(list)
        self.connection_graph = defaultdict(set)
        self.ledger = LOGS / 'third_brain_ledger.jsonl'
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        # Sharded ledger for faster lookup
        self.sharded_ledger = True
        # Sharded weave store
        from sharded_store import get_store
        self.store = get_store()

    def _load_recent(self, path, limit=500):
        items = []
        if not path.exists():
            return items
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        items.append(json.loads(line))
                    except Exception:
                        pass
                    if len(items) >= limit:
                        break
        except Exception:
            pass
        return items

    def _grade_item(self, item):
        # Active feedback grading with weights
        fact_score = 1.0 if item.get('confidence_separation') == 'KNOWN FACT' else 0.5
        pred_conf = item.get('predictor_confidence') or item.get('convergence',{}).get('predictor_conf') or 0.5
        conv_stab = item.get('convergence',{}).get('stability') or item.get('stability') or 0.5
        novelty = random.uniform(0.3,0.9)  # placeholder for novelty detection
        feedback = item.get('reward', 0.5)
        # Weave lattice influence from bias-variance controller
        bv = item.get('bias_variance') or {}
        weave_weight = item.get('bias_variance_weave_weight')
        if weave_weight is None:
            # fallback to bias_variance adjusted_score
            weave_weight = 1.0 + float(bv.get('reward', 0.0))
        # Normalize weave weight to 0-1 range for grading
        weave_influence = min(1.0, max(0.0, (weave_weight - 0.1) / 2.0))
        score = (
            self.weights['fact_consistency'] * fact_score +
            self.weights['predictor_confidence'] * pred_conf +
            self.weights['convergence_stability'] * conv_stab +
            self.weights['novelty'] * novelty +
            self.weights['feedback_signal'] * feedback +
            0.05 * weave_influence  # small lattice boost
        )
        return round(score,4)

    def _info_braid(self, a, b):
        # Braid two info items into a composite representation
        # Simple heuristic: merge keys, average numeric scores
        braided = {}
        for k in set(a.keys()) | set(b.keys()):
            va = a.get(k)
            vb = b.get(k)
            if isinstance(va, (int,float)) and isinstance(vb,(int,float)):
                braided[k] = (va+vb)/2
            else:
                braided[k] = va or vb
        braided['_braided'] = True
        return braided

    def _make_connections(self, items):
        # Build support vectors with deltas and connection graph
        for i, it in enumerate(items):
            key = it.get('query') or it.get('query_original') or str(i)
            # support vector with delta
            weave_weight = it.get('bias_variance_weave_weight')
            bv = it.get('bias_variance') or {}
            vec = {
                'timestamp': time.time(),
                'grade': self._grade_item(it),
                'source': it.get('response_source'),
                'weave_weight': weave_weight if weave_weight is not None else 1.0,
                'bias_variance_reward': bv.get('reward', 0.0),
                'flexibility': bv.get('flexibility', 3.0)
            }
            self.support_vectors[key].append(vec)
            # Keep last 10 per key
            if len(self.support_vectors[key]) > 10:
                self.support_vectors[key] = self.support_vectors[key][-10:]
            # delta between consecutive vectors
            if len(self.support_vectors[key]) > 1:
                prev = self.support_vectors[key][-2]
                delta = vec['grade'] - prev['grade']
                vec['delta'] = round(delta,4)
            else:
                vec['delta'] = 0.0
            # info braiding with neighbor
            if i > 0:
                braided = self._info_braid(it, items[i-1])
                self.connection_graph[key].add(items[i-1].get('query') or str(i-1))
        # Log ledger
        entry = {
            'ts': time.time(),
            'items_processed': len(items),
            'avg_grade': round(sum(self._grade_item(it) for it in items)/max(1,len(items)),4),
            'support_vector_keys': len(self.support_vectors),
            'connections': sum(len(v) for v in self.connection_graph.values())
        }
        # Write to sharded ledger for faster lookup
        try:
            sys_path = Path(__file__).resolve().parent
            import sys
            sys.path.insert(0, str(sys_path))
            from sharded_ledger import append as shard_append
            shard_append(entry)
        except Exception:
            # Fallback to old ledger
            with open(self.ledger, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry)+'\n')

    def learn_loop_step(self):
        # Pull data from storage
        all_items = []
        for p in STORAGE_PATHS:
            all_items.extend(self._load_recent(p, limit=200))
        if not all_items:
            return {'status':'no_data'}
        # Grade and connect
        self._make_connections(all_items)
        # Update weights via active feedback: simple momentum
        # If average grade is high, increase fact weight slightly
        avg = sum(self._grade_item(it) for it in all_items)/max(1,len(all_items))
        # Adaptive weight shift
        shift = 0.01
        if avg > 0.75:
            self.weights['fact_consistency'] = min(0.5, self.weights['fact_consistency']+shift)
        else:
            self.weights['fact_consistency'] = max(0.2, self.weights['fact_consistency']-shift)
        # Normalize
        total = sum(self.weights.values())
        for k in self.weights:
            self.weights[k] /= total
        # Persist weights
        try:
            self.weights_path.write_text(json.dumps(self.weights))
        except Exception:
            pass
        return {
            'status':'learned',
            'items': len(all_items),
            'avg_grade': round(avg,4),
            'weights': self.weights,
            'support_vectors': len(self.support_vectors)
        }

    def run_continuous(self, interval=60):
        while True:
            res = self.learn_loop_step()
            time.sleep(interval)

def get_third_brain():
    return ThirdBrain()

if __name__ == '__main__':
    tb = get_third_brain()
    print(tb.learn_loop_step())
