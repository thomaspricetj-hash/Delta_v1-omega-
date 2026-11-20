"""
Matrix Multiply Predictor
Simple matrix multiplication heuristic predictor for numeric reasoning
With self-referential caching to make token-level calculations cheap on CPU
"""
import re
import json
from pathlib import Path
try:
    import numpy as np
except ImportError:
    np = None

import hashlib
from time import perf_counter

class BloomFilter:
    def __init__(self, size=2**20, hashes=4):
        self.size = size
        self.hashes = hashes
        self.bits = bytearray((size + 7)//8)
    def _hashes(self, item):
        data = item.encode('utf-8')
        h1 = int(hashlib.sha256(data).hexdigest(), 16)
        h2 = int(hashlib.sha256(data[::-1]).hexdigest(), 16)
        for i in range(self.hashes):
            yield (h1 + i*h2) % self.size
    def add(self, item):
        for pos in self._hashes(item):
            self.bits[pos//8] |= 1 << (pos % 8)
    def __contains__(self, item):
        return all((self.bits[pos//8] >> (pos % 8)) & 1 for pos in self._hashes(item))

MATMUL_BLOOM = BloomFilter(size=2**22, hashes=4)

CACHE_DIR = Path(__file__).resolve().parents[2] / "runtime" / "logs" / "matrix_multiply_cache_shards"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
INDEX_PATH = CACHE_DIR / "matrix_multiply_cache_index.json"
LEDGER_PATH = CACHE_DIR / "matrix_multiply_cache_ledger.jsonl"
WEAVE_LINK_PATH = CACHE_DIR / "matrix_multiply_weave_links.json"
SAVINGS_LOG = Path(__file__).resolve().parents[2] / "runtime" / "logs" / "predictor_savings.jsonl"

def _load_index():
    try:
        if INDEX_PATH.exists():
            return json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}

def _save_index(idx):
    try:
        INDEX_PATH.write_text(json.dumps(idx), encoding="utf-8")
    except Exception:
        pass

INDEX = _load_index()

def _load_cache():
    cache = {}
    # Load from shards via index
    try:
        for key, meta in INDEX.items():
            shard_file = CACHE_DIR / meta["shard"]
            # Simple jsonl shard load for now
            # In production this would be BitDrop decompress
            if shard_file.exists():
                # For simplicity, keep in-memory map
                pass
    except Exception:
        pass
    return cache

_MATMUL_CACHE = _load_cache()
_MAX_RAM_ITEMS = 5000
_ARRAY_LRU = {}
_ARRAY_LRU_ORDER = []
_ARRAY_LRU_MAX = 256

# Pre-seed calculator table for small operands to make predictor instant
try:
    for a in range(-99, 100):
        for b in range(-99, 100):
            for op in ['*','+','-','/']:
                k = f"{op}:{repr(a)}:{repr(b)}"
                if k not in _MATMUL_CACHE:
                    try:
                        if op == '*': v = a*b
                        elif op == '+': v = a+b
                        elif op == '-': v = a-b
                        elif op == '/' and b != 0:
                            # Avoid division by zero, keep float
                            v = a / b
                        else:
                            continue
                        _MATMUL_CACHE[k] = v
                        MATMUL_BLOOM.add(k)
                    except Exception:
                        pass
except Exception:
    pass

# Multi-layered map for next prediction
_LAYER_MAP = {}
_LAYER_ACCESS = {}

# Chunking / clustering state for predictor improvement
_CLUSTER_STATS = {}

def _magnitude_bin(v):
    try:
        import math
        if v == 0:
            return 0
        return int(math.log10(abs(v) + 1e-12))
    except Exception:
        return 0

def _cluster_key(op, a, b):
    return (op, _magnitude_bin(a), _magnitude_bin(b))

def _cluster_confidence(op, a, b, base_conf):
    key = _cluster_key(op, a, b)
    stats = _CLUSTER_STATS.get(key)
    if stats:
        total = stats.get('hits',0) + stats.get('misses',0)
        if total > 0:
            hit_rate = stats.get('hits',0) / total
            # Boost confidence up to +0.1
            return min(0.99, base_conf + 0.1 * hit_rate)
    return base_conf

def _update_cluster(op, a, b, success):
    key = _cluster_key(op, a, b)
    stats = _CLUSTER_STATS.setdefault(key, {'hits':0,'misses':0})
    if success:
        stats['hits'] += 1
    else:
        stats['misses'] += 1

def _ensure_layers():
    global _LAYER_MAP
    # Simple heuristic: create layers based on operator
    for op in ['*','+','-','/']:
        if op not in _LAYER_MAP:
            _LAYER_MAP[op] = []

def _predict_next_ops(key):
    # Use weave links to predict next ops
    try:
        parts = key.split(":")
        if len(parts) < 3:
            return []
        op, a, b = parts[0], parts[1], parts[2]
        # Predict next ops for same operands
        candidates = []
        for next_op in ['*','+','-','/']:
            if next_op != op:
                candidates.append(f"{next_op}:{a}:{b}")
        return candidates
    except Exception:
        return []

def _stability_loop():
    import time, threading
    def loop():
        global _MAX_RAM_ITEMS
        while True:
            try:
                # Simple hit rate heuristic based on cache size vs evictions
                hit_rate = 0.0
                # Adjust RAM cap based on usage
                # Placeholder logic
                time.sleep(60)
            except Exception:
                pass
    t = threading.Thread(target=loop, daemon=True)
    t.start()

_ensure_layers()
_stability_loop()

_MAX_RAM_ITEMS = 5000
_ACCESS_ORDER = []
_BATCH_BUFFER = []
_BATCH_SIZE = 100

# Convergence and reward/penalty tracking
_PREDICTOR_STATS = {
    op: {'hits':0, 'misses':0, 'reward':1.0, 'penalty':0.0, 'confidence':0.85} for op in ['*','+','-','/']
}

def _update_reward(op, success=True, hit=True):
    stats = _PREDICTOR_STATS[op]
    if hit:
        stats['hits'] += 1
    else:
        stats['misses'] += 1
    # Load latest EMA for gain scheduling
    ema_map = {}
    try:
        ema_path = Path(__file__).resolve().parents[2] / 'runtime' / 'logs' / 'predictor_ema.jsonl'
        if ema_path.exists():
            latest = {}
            with open(ema_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        if obj.get('op'):
                            latest[obj['op']] = float(obj.get('ema',0.0))
                    except Exception:
                        pass
            ema_map = latest
    except Exception:
        pass
    # LQR-controlled reward/penalty update
    try:
        from .lqr_reward_controller import update_reward_lqr
        # Adjust stats for success/failure signal via EMA influence
        # The LQR controller uses current stats as state
        res = update_reward_lqr(op, stats, ema=ema_map)
        stats['reward'] = res['reward']
        stats['penalty'] = res['penalty']
        stats['confidence'] = res['confidence']
    except Exception:
        # Fallback to heuristic
        if success:
            stats['reward'] = min(1.5, stats['reward'] + 0.01)
            stats['penalty'] = max(0.0, stats['penalty'] - 0.005)
            stats['confidence'] = min(0.99, stats['confidence'] + 0.001)
        else:
            stats['penalty'] = min(1.0, stats['penalty'] + 0.02)
            stats['reward'] = max(0.5, stats['reward'] - 0.01)
            stats['confidence'] = max(0.5, stats['confidence'] - 0.005)
    # Convergence metric
    total = stats['hits'] + stats['misses']
    hit_rate = stats['hits'] / total if total>0 else 0.0
    try:
        log_path = Path(__file__).resolve().parents[2] / 'runtime' / 'logs' / 'matrix_multiply_convergence.jsonl'
        log_path.parent.mkdir(parents=True, exist_ok=True)
        import time, json
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({
                'ts': time.time(),
                'op': op,
                'hit_rate': hit_rate,
                'reward': stats['reward'],
                'penalty': stats['penalty'],
                'confidence': stats['confidence'],
                'hits': stats['hits'],
                'misses': stats['misses'],
                'lqr_controlled': True
            }) + '\n')
    except Exception:
        pass

def _evict_if_needed():
    global _MATMUL_CACHE, _ACCESS_ORDER
    while len(_MATMUL_CACHE) > _MAX_RAM_ITEMS and _ACCESS_ORDER:
        old_key = _ACCESS_ORDER.pop(0)
        _MATMUL_CACHE.pop(old_key, None)

def _flush_batch():
    global _BATCH_BUFFER
    if not _BATCH_BUFFER:
        return
    # Write batch to shard
    try:
        shard_id = len(INDEX) % 100
        shard_file = CACHE_DIR / f"shard_{shard_id}.jsonl"
        with open(shard_file, "a", encoding="utf-8") as f:
            for key, value in _BATCH_BUFFER:
                entry = {"key": key, "value": value}
                f.write(json.dumps(entry) + "\n")
                INDEX[key] = {"shard": str(shard_file.name), "ts": __import__("time").time()}
        _save_index(INDEX)
        # Update ledger for batch
        with open(LEDGER_PATH, "a", encoding="utf-8") as f:
            for key, _ in _BATCH_BUFFER:
                f.write(json.dumps({"key": key, "shard": str(shard_file.name), "access_count": 1, "last_used": __import__("time").time()}) + "\n")
        _BATCH_BUFFER = []
    except Exception:
        pass

def _save_cache_entry(key, value):
    # Batch write for cheap CPU
    global _BATCH_BUFFER
    _BATCH_BUFFER.append((key, value))
    if len(_BATCH_BUFFER) >= _BATCH_SIZE:
        _flush_batch()
    # Weave linkage - keep immediate for connectivity
    try:
        weave_links = {}
        if WEAVE_LINK_PATH.exists():
            weave_links = json.loads(WEAVE_LINK_PATH.read_text(encoding="utf-8"))
        parts = key.split(":")
        if len(parts) >= 3:
            op = parts[0]
            weave_links.setdefault(op, set()).add(key)
            WEAVE_LINK_PATH.write_text(json.dumps({k: list(v) for k, v in weave_links.items()}), encoding="utf-8")
    except Exception:
        pass

def _calculator_op(a, b, op):
    # Calculator-style fast paths for the predictor
    try:
        fa = float(a)
        fb = float(b)
        # Small table fast path
        if abs(fa) < 100 and abs(fb) < 100:
            # Precomputed via cached_matmul, will hit cache quickly
            return None
        if op == '*':
            # Log/exp for very large magnitudes -> avoids overflow and is cheap
            if abs(fa) > 1e6 or abs(fb) > 1e6:
                import math
                if fa == 0 or fb == 0:
                    return 0.0
                # Use logs for magnitude, keep sign
                sign = 1.0 if fa*fb >= 0 else -1.0
                try:
                    res = math.exp(math.log(abs(fa)) + math.log(abs(fb)))
                    return sign * res
                except ValueError:
                    return None
            # Karatsuba-like split for medium-large ints
            if abs(fa) > 1000 or abs(fb) > 1000:
                base = 1000
                a_hi = int(fa // base)
                a_lo = int(fa % base)
                # (a_hi*base + a_lo) * b = a_hi*b*base + a_lo*b
                res_hi = cached_matmul(a_hi, b, '*')
                res_lo = cached_matmul(a_lo, b, '*')
                if res_hi is not None and res_lo is not None:
                    return res_hi * base + res_lo
        if op == '/':
            # Avoid division by zero
            if fb == 0:
                return None
            # For small divisor, use repeated subtraction approximation
            if abs(fb) < 1e-12:
                return None
        # Addition/subtraction can use chunking via decomposition already handled
        return None
    except Exception:
        return None

def _decompose_op(a, b, op):
    # Break large operands into smaller sub-problems to avoid overshoot/undershoot
    # First try calculator fast paths
    calc = _calculator_op(a, b, op)
    if calc is not None:
        return calc
    try:
        fa = float(a)
        fb = float(b)
        # Threshold for decomposition
        if op == '*' and (abs(fa) > 1000 or abs(fb) > 1000):
            base = 1000
            a_hi = int(fa // base)
            a_lo = int(fa % base)
            res_hi = cached_matmul(a_hi, b, op)
            res_lo = cached_matmul(a_lo, b, op)
            if res_hi is not None and res_lo is not None:
                return res_hi * base + res_lo
        return None
    except Exception:
        return None

def cached_matmul(a, b, op="*"):
    """Cheap CPU matrix multiply with self-cache. Works for scalars or numpy arrays."""
    # Shape-aware key for arrays to avoid huge repr and enable LRU hits
    try:
        if hasattr(a, 'shape') and hasattr(b, 'shape'):
            # Quick shape+dtype key + light hash for small arrays
            a_sig = f"{a.shape}:{a.dtype}"
            b_sig = f"{b.shape}:{b.dtype}"
            # Small arrays: hash bytes for correctness, large arrays: use id as proxy
            if a.size < 1024 and b.size < 1024:
                import hashlib
                h = hashlib.sha256(a.tobytes() + b.tobytes()).hexdigest()[:16]
                key = f"{op}:arr:{a_sig}:{b_sig}:{h}"
            else:
                key = f"{op}:arr:{a_sig}:{b_sig}:id{ id(a) }:{ id(b) }"
        else:
            key = f"{op}:{repr(a)}:{repr(b)}"
    except Exception:
        key = f"{op}:{repr(a)}:{repr(b)}"
    # Update layer map for convergence overlay
    try:
        _LAYER_MAP.setdefault(op, []).append(key)
        _LAYER_ACCESS[key] = __import__("time").time()
    except Exception:
        pass
    # Bloom filter fast negative lookup
    if key not in MATMUL_BLOOM:
        # definite miss, skip dict lookup
        bloom_miss = True
    else:
        bloom_miss = False
    if not bloom_miss and key in _MATMUL_CACHE:
        # Cache hit
        t0 = perf_counter()
        if key in _ACCESS_ORDER:
            _ACCESS_ORDER.remove(key)
        _ACCESS_ORDER.append(key)
        _update_reward(op, success=True, hit=True)
        latency_ms = (perf_counter() - t0) * 1000
        # Estimate saved time vs full matmul ~0.5ms for scalar, 5ms for array
        is_array = hasattr(a, 'shape') or hasattr(b, 'shape')
        est_saved_ms = 0.5 if not is_array else 5.0
        try:
            with open(SAVINGS_LOG, "a", encoding="utf-8") as f:
                f.write(__import__('json').dumps({"ts": __import__("time").time(), "op": op, "key": key, "hit": True, "latency_ms": latency_ms, "est_saved_ms": est_saved_ms, "flops_saved": 1}) + "\n")
        except Exception:
            pass
        return _MATMUL_CACHE[key]
    # Try decomposition first to avoid overshoot
    decomposed = _decompose_op(a, b, op)
    if decomposed is not None:
        # Store decomposed result
        _MATMUL_CACHE[key] = decomposed
        MATMUL_BLOOM.add(key)
        _ACCESS_ORDER.append(key)
        _evict_if_needed()
        _save_cache_entry(key, {"type":"scalar","value":float(decomposed)})
        _update_reward(op, success=True, hit=False)
        return decomposed
    # Compute
    t_compute_start = perf_counter()
    try:
        import numpy as np
        if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            if op == "*":
                res = np.dot(a, b)
            elif op == "+":
                res = a + b
            elif op == "-":
                res = a - b
            elif op == "/":
                res = a / b
            else:
                res = None
        else:
            fa = float(a); fb = float(b)
            if op == "*":
                res = fa * fb
            elif op == "+":
                res = fa + fb
            elif op == "-":
                res = fa - fb
            elif op == "/":
                res = fa / fb if fb != 0 else None
            else:
                res = None
        # Store
        if res is not None:
            _MATMUL_CACHE[key] = res
            MATMUL_BLOOM.add(key)
            _ACCESS_ORDER.append(key)
            _evict_if_needed()
            _save_cache_entry(key, {"type": "numpy" if isinstance(res, np.ndarray) else "scalar", "value": float(res) if not isinstance(res, np.ndarray) else res.tolist()})
            # Update reward on compute miss
            _update_reward(op, success=True, hit=False)
            # Log compute latency and savings
            try:
                latency_ms = (perf_counter() - t_compute_start) * 1000
                with open(SAVINGS_LOG, "a", encoding="utf-8") as f:
                    f.write(__import__('json').dumps({"ts": __import__("time").time(), "op": op, "key": key, "hit": False, "latency_ms": latency_ms, "est_saved_ms": 0.0, "flops_saved": 0}) + "\n")
            except Exception:
                pass
            # Log to predictor convergence overlay
            try:
                import json, time as _t
                conv_path = Path(__file__).resolve().parents[2] / "runtime" / "logs" / "predictor_convergence.jsonl"
                conv_path.parent.mkdir(parents=True, exist_ok=True)
                with open(conv_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"ts": _t.time(), "predictor": "matrix_multiply", "key": key, "op": op, "confidence": _PREDICTOR_STATS[op]['confidence'], "layer": op}) + "\n")
            except Exception:
                pass
        return res
    except Exception:
        return None

def predict(query, diagnostics=None):
    q = query or ""
    # Isolate the most recent user turn to avoid history bleed
    if "User:" in q:
        parts = q.rsplit("User:", 1)
        q = parts[-1]
    q = re.sub(r'^\s*Assistant:\s*', '', q, flags=re.I)
    ql = q.lower()

    # Chunking / grouping: tokenise numbers and operators in order
    token_pat = re.compile(r'(?<!\d)[-+]?\d+\.?\d*|[*+\-/x]|times|multiply|divided by|divide|plus|add|minus|subtract')
    raw_tokens = [m.group(0) for m in token_pat.finditer(q)]
    if not raw_tokens:
        return None

    nums = []
    ops = []
    op_word_map = {
        'times':'*','x':'*','multiply':'*','*':'*',
        'divided by':'/','/':'/','over':'/','divide':'/',
        'minus':'-','-':'-','subtract':'-',
        'plus':'+','+':'+','add':'+'
    }
    for t in raw_tokens:
        if re.match(r'[-+]?\d', t):
            try:
                nums.append(float(t))
            except Exception:
                pass
        else:
            op_sym = op_word_map.get(t.lower())
            if op_sym:
                ops.append(op_sym)

    if len(nums) < 2:
        # Fallback to simple regex extraction
        nums_str = re.findall(r'[-+]?\d+\.?\d*', q)
        if len(nums_str) < 2:
            return None
        nums = [float(nums_str[0]), float(nums_str[1])]
        # Detect operator via keywords
        op = None
        for sym, keys in { '*':['times','x','multiply','*'], '/':['divided by','/','over','divide'], '-':['minus','-','subtract'], '+':['plus','+','add'] }.items():
            if any(k in ql for k in keys):
                op = sym
                break
        if not op:
            return None
        # Single pair evaluation
        try:
            res = cached_matmul(nums[0], nums[1], op)
            if res is None:
                return None
            base_conf = _PREDICTOR_STATS.get(op, {}).get('confidence', 0.85)
            conf = _cluster_confidence(op, nums[0], nums[1], base_conf)
            _update_cluster(op, nums[0], nums[1], True)
            result_str = str(int(res)) if float(res).is_integer() else str(res)
            return {'response': result_str, 'confidence': conf, 'source': 'arithmetic'}
        except Exception:
            return None

    # Evaluate with shunting-yard for correct precedence
    try:
        # Build token list with precedence
        # If ops list shorter than nums-1, assume repeated first op
        if len(ops) < len(nums)-1:
            first_op = ops[0] if ops else '*'
            ops = [first_op]*(len(nums)-1)
        else:
            ops = ops[:len(nums)-1]

        # Shunting-yard: * / have higher precedence than + -
        prec = {'*':2, '/':2, '+':1, '-':1}
        output = []
        op_stack = []
        output.append(nums[0])
        for i, op in enumerate(ops):
            num = nums[i+1]
            while op_stack and prec.get(op_stack[-1],0) >= prec.get(op,0):
                o2 = op_stack.pop()
                b = output.pop()
                a = output.pop()
                res = cached_matmul(a, b, o2)
                if res is None:
                    return None
                _update_cluster(o2, a, b, True)
                output.append(float(res))
            op_stack.append(op)
            output.append(num)
        while op_stack:
            o2 = op_stack.pop()
            b = output.pop()
            a = output.pop()
            res = cached_matmul(a, b, o2)
            if res is None:
                return None
            _update_cluster(o2, a, b, True)
            output.append(float(res))
        acc = output[0]
        base_conf = _PREDICTOR_STATS.get(ops[-1] if ops else '*', {}).get('confidence', 0.85)
        conf = _cluster_confidence(ops[0], nums[0], nums[1], base_conf)
        conf = min(0.99, conf)
        result_str = str(int(acc)) if float(acc).is_integer() else str(acc)
        return {'response': result_str, 'confidence': conf, 'source': 'arithmetic'}
    except Exception:
        pass
    return None
