"""
Abbreviation map for Translation OS
Learn short codes for frequent words/phrases, persist via BitDrop v2
"""
from pathlib import Path
import sys, json, re
from collections import Counter

BASE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE / "scripts"))
from bitdrop_persistence import save_json, load_json_bitdrop

MAP_PATH = BASE / "runtime" / "logs" / "abbrev_map.json"
COUNTER_PATH = BASE / "runtime" / "logs" / "abbrev_counter.json"

STOPWORDS = set("""
a an the and or but if then else for with without in on at to from by is are was were be been being
of to the it its that this these those
""".split())

def _load_map():
    data = load_json_bitdrop(MAP_PATH)
    if not data:
        return {}, {}
    fwd = data.get("fwd", {})
    rev = data.get("rev", {})
    return fwd, rev

def _save_map(fwd, rev):
    save_json(MAP_PATH, {"fwd": fwd, "rev": rev})

def _tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

def learn_from_texts(texts, min_freq=3):
    fwd, rev = _load_map()
    counter_data = load_json_bitdrop(COUNTER_PATH) or {}
    # Update counts - normalize keys to be space-free
    def norm_key(k):
        return re.sub(r'\s+', '', k.lower())
    for t in texts:
        toks = _tokenize(t)
        for w in toks:
            if w in STOPWORDS or len(w) < 3:
                continue
            key = norm_key(w)
            counter_data[key] = counter_data.get(key, 0) + 1
        for i in range(len(toks)-1):
            bg = f"{toks[i]} {toks[i+1]}"
            if toks[i] in STOPWORDS:
                continue
            key = norm_key(bg)
            counter_data[key] = counter_data.get(key, 0) + 1
        for i in range(len(toks)-2):
            tg = f"{toks[i]} {toks[i+1]} {toks[i+2]}"
            key = norm_key(tg)
            counter_data[key] = counter_data.get(key, 0) + 1
    save_json(COUNTER_PATH, counter_data)
    new_mappings = 0
    for phrase, freq in counter_data.items():
        if freq < min_freq:
            continue
        # Normalize phrase for internal storage: lowercase, no spaces
        phrase_norm = re.sub(r'\s+', '', phrase.lower())
        if phrase_norm in fwd:
            continue
        code = "".join([w[0].upper() for w in phrase.split()[:3]])
        base = code
        i = 1
        while code in rev:
            code = f"{base}{i}"
            i += 1
        fwd[phrase_norm] = code
        rev[code] = phrase
        new_mappings += 1
    if new_mappings > 0:
        _save_map(fwd, rev)
        try:
            sys.path.insert(0, str(BASE / "infrastructure"))
            from master_map_integrator import integrate_all
            integrate_all()
        except Exception:
            pass
    return fwd, rev

def compress(text):
    fwd, _ = _load_map()
    if not fwd:
        return text
    # Work on space-insensitive matching
    out = text
    # Sort by original phrase length for greedy matching
    items = sorted(fwd.items(), key=lambda x: len(x[0]), reverse=True)
    for ph_norm, code in items:
        # Reconstruct spaced version for regex
        # ph_norm is space-free, we need to match with optional spaces
        # Build pattern that allows spaces between words
        words = re.split(r'[^a-z0-9]+', ph_norm)
        if not words:
            continue
        pattern_words = [re.escape(w) for w in words if w]
        if not pattern_words:
            continue
        pattern = r'(?i)\b' + r'\s+'.join(pattern_words) + r'\b'
        out = re.sub(pattern, code, out)
    return out

def expand(text):
    _, rev = _load_map()
    if not rev:
        return text
    out = text
    for code, phrase in rev.items():
        pattern = r"\b" + re.escape(code) + r"\b"
        out = re.sub(pattern, phrase, out)
    return out

def get_map():
    return _load_map()

def learn_from_master_map(min_freq=3):
    """Extract texts from master map layers and learn abbreviations from them."""
    try:
        sys.path.insert(0, str(BASE / "infrastructure" / "foldable_map"))
        from foldable_map.foldable_map import _foldable_map
        texts = []
        for layer in _foldable_map.layers:
            for key, val in layer.items():
                # collect string values
                if isinstance(val, str):
                    texts.append(val)
                elif isinstance(val, dict):
                    # flatten dict values
                    for v in val.values():
                        if isinstance(v, str):
                            texts.append(v)
        if texts:
            learn_from_texts(texts, min_freq=min_freq)
    except Exception:
        pass

if __name__ == "__main__":
    samples = ["What is a dog", "What is a cat", "What is the capital of France"]
    learn_from_texts(samples, min_freq=1)
    s = "What is a dog and What is a cat"
    print(compress(s))
    samples = ["What is a dog", "What is a cat", "What is the capital of France"]
    learn_from_texts(samples, min_freq=1)
    s = "What is a dog and What is a cat"
    print(compress(s))
