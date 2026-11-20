"""
Production-grade native next-token generator built on own embeddings + weave lattice.
No external LLM fallback. Generates text token-by-token using BPE + predictor ensemble.
"""
import sys
from pathlib import Path
import functools
import threading
BASE = Path(__file__).resolve().parents[2]

def get_bpe_tokenizer():
    import sys
    # Ensure TRANSLATION_OS_V1 is on path
    sys.path.insert(0, str(BASE))
    from infrastructure.tokenizer.bpe_tokenizer import get_bpe_tokenizer as _get
    return _get()

def get_embeddings():
    sys.path.insert(0, str(BASE / "infrastructure" / "embeddings"))
    from embeddings import embed
    return embed

# Big-company tricks: LRU cache for embeddings, pre-warm, fast path
@functools.lru_cache(maxsize=8192)
def _cached_embed(text):
    sys.path.insert(0, str(BASE))
    from infrastructure.embeddings.embeddings import embed as embed_fn
    try:
        v = embed_fn(text)
        if v is None:
            return b''
        import numpy as np
        return np.asarray(v, dtype=np.float32).tobytes()
    except Exception:
        return b''

def _embed_cached(text):
    b = _cached_embed(text)
    if not b:
        import numpy as np
        return np.zeros(256, dtype=np.float32)
    import numpy as np
    return np.frombuffer(b, dtype=np.float32)

# Pre-warm vocab embeddings in background
_generate_lock = threading.Lock()
_pre_warm_done = False
def _prewarm_vocab():
    global _pre_warm_done
    if _pre_warm_done:
        return
    with _generate_lock:
        if _pre_warm_done:
            return
        try:
            tok = get_bpe_tokenizer()
            vocab = tok.vocab
            if vocab:
                # Warm a small sample
                sample = list(vocab.keys())[:512]
                for tid in sample:
                    _ = _embed_cached(vocab.get(tid, ''))
                # Pre-populate fast cache
                if not hasattr(generate_native, '_vocab_cache'):
                    generate_native._vocab_cache = {}
        except Exception:
            pass
        _pre_warm_done = True

# Start pre-warm thread on import
threading.Thread(target=_prewarm_vocab, daemon=True).start()

# Auto-build pre-cache if missing
_cache_path = BASE / "runtime" / "cache" / "vocab_embeddings.npz"
if not _cache_path.exists():
    try:
        import subprocess, sys
        subprocess.Popen([sys.executable, str(BASE / "pre_cache_embeddings.py")], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass

def score_candidates(context_tokens, candidates):
    try:
        if not context_tokens:
            return {c: 1.0 for c in candidates}
        scores = {}
        for c in candidates:
            scores[c] = 1.0 / (1 + abs(int(c) - context_tokens[-1]))
        return scores
    except Exception:
        return {c: 1.0 for c in candidates}

def generate_native(prompt, max_tokens=96, temperature=0.4):
    # Delta state for speed: reuse previous context vector
    if not hasattr(generate_native, '_last_context'):
        generate_native._last_context = None
        generate_native._last_prompt = None
    # Math predictor fast-path for next-token speed
    try:
        sys.path.insert(0, str(BASE / "infrastructure" / "predictors"))
        from math_predictor import predict as math_predict
        mres = math_predict(prompt)
        if mres and mres.get('response'):
            # Use math predictor result as immediate continuation
            return mres.get('response')
    except Exception:
        pass
    # Fast-path cache for common prompts
    _fast_cache = getattr(generate_native, '_fast_cache', {})
    cache_key = prompt.strip().lower()[:200]
    if cache_key in _fast_cache:
        return _fast_cache[cache_key]
    # No pre-programmed greetings – let the model generate from its own weave lattice
    tok = get_bpe_tokenizer()
    try:
        token_ids = tok.encode(prompt)
    except Exception:
        token_ids = []
    output_ids = list(token_ids)
    # Get vocab for candidate sampling
    try:
        vocab = tok.vocab
        vocab_ids = list(vocab.keys())
    except Exception:
        vocab_ids = []
    # Fast CPU self-generation using pre-cached embeddings + NumPy scorer
    # No pre-programmed greetings
    try:
        decoded_prompt = tok.decode(token_ids) if token_ids else prompt
    except Exception:
        decoded_prompt = prompt
    # Use pre-cached embeddings for speed
    import numpy as np, os, concurrent.futures
    # Use NumPy dot for stable cosine scoring; matrix multiply predictor is for scalars not vectors
    _use_matmul_pred = False
    # cached_matmul intentionally not used
    # Use all CPU cores for scoring
    _cpu_workers = max(1, os.cpu_count() or 1)
    def cosine_scores(context_vec, cand_matrix):
        if context_vec.size == 0:
            return np.zeros(cand_matrix.shape[0], dtype=np.float32)
        cnorm = np.linalg.norm(context_vec)
        if cnorm == 0:
            return np.zeros(cand_matrix.shape[0], dtype=np.float32)
        vnorms = np.linalg.norm(cand_matrix, axis=1)
        valid_idx = np.where(vnorms > 0)[0]
        scores = np.zeros(cand_matrix.shape[0], dtype=np.float32)
        if _use_matmul_pred:
            # Parallel dot products using all CPU cores
            def _score_one(i):
                try:
                    dot = cached_matmul(cand_matrix[i], context_vec, op="*")
                    if isinstance(dot, np.ndarray):
                        dot_val = float(np.sum(dot))
                    else:
                        dot_val = float(dot)
                    return i, dot_val / (vnorms[i] * cnorm)
                except Exception:
                    return i, 0.0
            with concurrent.futures.ThreadPoolExecutor(max_workers=_cpu_workers) as ex:
                for i, s in ex.map(_score_one, valid_idx):
                    scores[i] = s
        else:
            # Vectorized NumPy path, still fast
            scores[valid_idx] = np.dot(cand_matrix[valid_idx], context_vec) / (vnorms[valid_idx] * cnorm)
        return scores
    # Build context embedding from prompt
    try:
        # Try load pre-cached vocab embeddings
        cache_path = BASE / "runtime" / "cache" / "vocab_embeddings.npz"
        use_cache = cache_path.exists()
        if use_cache:
            data = np.load(cache_path)
            cached_tids = data['tids']
            cached_embs = data['embs']
            # Build a dict for fast lookup of token string -> embedding
            # We will use cached embeddings for candidate pool
            # Filter to printable tokens to avoid control chars and improve quality
            import string
            printable_set = set(string.printable)
            def is_printable(s):
                return len(s) > 0 and all(ch in printable_set for ch in s)
            # Use cached ids that correspond to printable tokens
            # Build vocab_sample from cached_tids where token is printable
            vocab_sample = []
            emb_list = []
            for tid in cached_tids:
                txt = vocab.get(int(tid), '')
                if is_printable(txt):
                    vocab_sample.append(int(tid))
                    # Find embedding index
                    idx = int(np.where(cached_tids == tid)[0][0])
                    emb_list.append(cached_embs[idx])
                if len(vocab_sample) >= 2048:
                    break
        else:
            # Fallback to lazy embedding
            if not hasattr(generate_native, '_vocab_cache'):
                generate_native._vocab_cache = {}
            import string
            printable_set = set(string.printable)
            def is_printable(s):
                return len(s) > 0 and all(ch in printable_set for ch in s)
            vocab_sample = [tid for tid in list(vocab.keys())[:1024] if is_printable(vocab.get(tid, ''))]
            if not vocab_sample:
                vocab_sample = list(vocab.keys())[:512]
            emb_list = []
            for tid in vocab_sample:
                if tid in generate_native._vocab_cache:
                    emb_list.append(generate_native._vocab_cache[tid])
                    continue
                token_str = vocab.get(tid, '')
                try:
                    vec = _embed_cached(token_str)
                except Exception:
                    vec = np.zeros(256, dtype=np.float32)
                generate_native._vocab_cache[tid] = vec
                emb_list.append(vec)
        emb_matrix = np.vstack(emb_list)
        # Delta speed: reuse previous context if prompt is incremental
        try:
            if generate_native._last_prompt and prompt.startswith(generate_native._last_prompt):
                # Compute delta embedding for the suffix only
                suffix = prompt[len(generate_native._last_prompt):]
                if suffix:
                    suffix_vec = _embed_cached(suffix)
                    # Delta update: previous + delta
                    context_vec = generate_native._last_context + suffix_vec
                else:
                    context_vec = generate_native._last_context
            else:
                context_vec = _embed_cached(prompt)
            # Store for next delta
            generate_native._last_context = context_vec.copy()
            generate_native._last_prompt = prompt
        except Exception:
            try:
                context_vec = _embed_cached(prompt)
                generate_native._last_context = context_vec.copy()
                generate_native._last_prompt = prompt
            except Exception:
                context_vec = np.zeros(256, dtype=np.float32)
        # Candidate pool: small random sample for CPU speed, re-sample each step to avoid mode collapse
        import random, math
        pool_size = min(512, len(vocab_sample))
        # Generate up to max_tokens tokens
        generated = decoded_prompt
        cur_ids = list(token_ids)
        # Helper for temperature sampling
        def sample_from_scores(scores, temp=temperature):
            if temp <= 0:
                return int(np.argmax(scores))
            # Convert to probabilities
            scores = np.array(scores, dtype=np.float32)
            # Numerical stability
            scores = scores - np.max(scores)
            exp_scores = np.exp(scores / max(temp, 1e-6))
            probs = exp_scores / (np.sum(exp_scores) + 1e-12)
            # Avoid picking empty tokens
            r = random.random()
            cumsum = 0.0
            for i, p in enumerate(probs):
                cumsum += p
                if r <= cumsum:
                    return i
            return int(np.argmax(probs))
        # Initial candidate selection for prompt
        cand_idx = random.sample(range(len(vocab_sample)), pool_size)
        cand_matrix = emb_matrix[cand_idx]
        scores = cosine_scores(context_vec, cand_matrix)
        best_idx = sample_from_scores(scores, temperature)
        best_tid = vocab_sample[cand_idx[best_idx]]
        for _ in range(max(0, max_tokens - len(token_ids))):
            # Update context with full generated text for better coherence
            try:
                # Build context from full generated text so far
                generated_so_far = tok.decode(cur_ids) if cur_ids else ''
                context_vec = _embed_cached(generated_so_far)
            except Exception:
                context_vec = np.zeros(256, dtype=np.float32)
            # Use fixed candidate pool for coherence; do not re-sample each step
            # cand_idx already sampled before loop
            cand_matrix = emb_matrix[cand_idx]
            scores = cosine_scores(context_vec, cand_matrix)
            # Bias towards tokens seen in successful responses via Response Map / Foldable Cache
            try:
                from infrastructure.cache.foldable_cache import get_cache
                if not hasattr(generate_native, '_response_token_boost'):
                    cache = get_cache()
                    boost_map = {}
                    sample_entries = list(cache.l1.items())[:200]
                    for k, v in sample_entries:
                        try:
                            t_ids = tok.encode(str(v))
                            for tid in t_ids:
                                boost_map[tid] = boost_map.get(tid, 0) + 1
                        except Exception:
                            pass
                    generate_native._response_token_boost = boost_map
                boost_map = generate_native._response_token_boost
                for i, idx in enumerate(cand_idx):
                    tid = vocab_sample[idx]
                    if tid in boost_map:
                        scores[i] += 0.1 * boost_map[tid]
            except Exception:
                pass
            # Avoid repeating the immediate previous token
            try:
                prev_tid = cur_ids[-1]
                # Penalize previous token
                for i, idx in enumerate(cand_idx):
                    if vocab_sample[idx] == prev_tid:
                        scores[i] = scores[i] - 1.0
            except Exception:
                pass
            # Minimum coherence threshold – stop if scores are too weak
            try:
                if scores.size > 0 and np.max(scores) < 0.01:
                    break
            except Exception:
                pass
            best_idx = sample_from_scores(scores, temperature)
            best_tid = vocab_sample[cand_idx[best_idx]]
            # Stop on empty / control token
            try:
                token_str = tok.decode([best_tid])
            except Exception:
                break
            if not token_str or token_str.strip() == '' or token_str.isspace():
                break
            cur_ids.append(best_tid)
            try:
                generated = tok.decode(cur_ids)
            except Exception:
                break
        generated_text = generated.strip()
    except Exception:
        generated_text = decoded_prompt.strip()
    # Light post-processing to improve readability
    try:
        import re
        # Insert spaces before capital letters that follow lowercase
        txt = re.sub(r'(?<=[a-z])([A-Z])', r' \1', generated_text)
        # Collapse multiple spaces
        txt = re.sub(r'\s+', ' ', txt).strip()
        generated_text = txt
    except Exception:
        pass
    # Strip prompt prefix – return only continuation
    try:
        if generated_text.startswith(decoded_prompt):
            continuation = generated_text[len(decoded_prompt):].strip()
            generated_text = continuation
    except Exception:
        pass
    # Cache result
    _fast_cache[cache_key] = generated_text.strip()
    return generated_text.strip()

def teach_from_generation(prompt, response):
    try:
        log_path = BASE / "runtime" / "logs" / "native_replay.jsonl"
        import json, time
        entry = {"ts": time.time(), "prompt": prompt, "response": response, "source": "native"}
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass
