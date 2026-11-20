"""
Hierarchical codebook tokenizer for Translation OS V1
Compresses input stream before transformer using layered codebook paths.
Optimizes for fewer tokens, not shorter characters.
"""
from pathlib import Path
import json
from collections import Counter
from typing import List, Tuple, Dict
import numpy as np
from .tree_store import TreeCodebook

class HierarchicalTokenCodebook:
    """
    Four-layer hierarchy:
    root -> common words -> phrases -> technical terms -> abbreviations
    Each layer is a TreeCodebook for fast lookup and small nodes.
    """
    def __init__(self, dir_path: str = "token_codebook", capacity_per_node: int = 1024):
        self.dir = Path(dir_path)
        self.dir.mkdir(parents=True, exist_ok=True)
        # One TreeCodebook per semantic layer
        self.layers = {
            "common": TreeCodebook(capacity_per_node, str(self.dir / "common")),
            "phrases": TreeCodebook(capacity_per_node, str(self.dir / "phrases")),
            "technical": TreeCodebook(capacity_per_node, str(self.dir / "technical")),
            "abbrev": TreeCodebook(capacity_per_node, str(self.dir / "abbrev")),
        }
        # Mapping from text -> (layer_name, code_id)
        self.text_to_ref: Dict[str, Tuple[str, int]] = {}
        self.ref_to_text: Dict[Tuple[str,int], str] = {}
        self.frozen = False
        # Simple LRU cache for encode
        self._encode_cache = {}
        self._encode_cache_max = 8192

    def _hash_text(self, text: str) -> float:
        # Stable float key for TreeCodebook, using hash
        import hashlib
        h = hashlib.sha256(text.encode()).digest()
        # map to float in [0,1)
        return float(int.from_bytes(h[:4], 'little')) / 2**32

    def add_entry(self, text: str, layer: str):
        if layer not in self.layers:
            raise ValueError(f"Unknown layer {layer}")
        key = self._hash_text(text)
        cb = self.layers[layer]
        # Ensure codebook knows this value
        if key not in cb.value_to_code:
            cb.add(key)
        code = cb.value_to_code[key]
        # Avoid overwriting existing ref mapping with a different text
        existing = self.ref_to_text.get((layer, code))
        if existing is not None and existing != text:
            # Collision: skip adding this text to codebook to preserve uniqueness
            # Optionally log, but keep existing mapping
            return
        self.text_to_ref[text] = (layer, code)
        self.ref_to_text[(layer, code)] = text

    def build_from_corpus(self, texts: List[str], min_count: int = 10, max_entries: int = 200000):
        # Chunking: collect n-grams up to 4
        ngram_counts = {n: Counter() for n in range(1,5)}
        for t in texts:
            words = t.lower().split()
            for n in range(1,5):
                for i in range(len(words)-n+1):
                    ngram = ' '.join(words[i:i+n])
                    ngram_counts[n][ngram] += 1
        # Grouping by first word to form semantic groups
        groups = {}
        for n, counter in ngram_counts.items():
            for phrase, c in counter.items():
                if c < min_count:
                    continue
                head = phrase.split()[0]
                groups.setdefault(head, []).append((phrase, c, n))
        # Clustering within groups using simple embedding similarity proxy
        # Use deterministic hash-based vector for clustering
        import hashlib
        def phrase_vec(p):
            h = hashlib.sha256(p.encode()).digest()
            # 8-dim float vector
            return np.array([int.from_bytes(h[i:i+4], 'little')/2**32 for i in range(0,32,4)], dtype=np.float32)
        # Assign layers with chunking + grouping + clustering + semantic divergence guard
        def cosine_sim(a,b):
            na = np.linalg.norm(a); nb = np.linalg.norm(b)
            if na==0 or nb==0: return 0.0
            return float(np.dot(a,b)/(na*nb))
        SEMANTIC_THRESHOLD = 0.75
        for head, items in groups.items():
            items.sort(key=lambda x: (-x[1], -x[2]))
            clusters = []
            for phrase, c, n in items[:200]:
                vec = phrase_vec(phrase)
                placed = False
                best_sim = 0.0
                for cluster in clusters:
                    centroid_vec, centroid_phrase = cluster[0]
                    sim = cosine_sim(vec, centroid_vec)
                    if sim >= SEMANTIC_THRESHOLD:
                        placed = True
                        best_sim = sim
                        break
                    if sim > best_sim:
                        best_sim = sim
                if not placed:
                    # Semantic divergence guard: if phrase is too divergent from existing clusters
                    # and frequency is low, skip adding to avoid polluting codebook
                    if len(clusters) > 0 and best_sim < SEMANTIC_THRESHOLD and c < min_count * 3:
                        # Divergent low-freq phrase under same head -> skip codebook entry
                        continue
                    clusters.append([(vec, phrase), (c,n)])
                # Decide layer
                if n == 1 and c >= min_count * 10:
                    layer = "common"
                elif n >= 3:
                    layer = "phrases"
                elif any(t in phrase for t in ["machine","learning","neural","transformer","database","sql"]):
                    layer = "technical"
                else:
                    layer = "phrases"
                if len(self.text_to_ref) < max_entries:
                    self.add_entry(phrase, layer)
        # Abbreviations
        import re
        abbrev_pattern = re.compile(r'^[A-Z]{2,}$')
        common_abbrevs = {"e.g.","i.e.","etc.","vs.","mr","mrs","dr","prof","st","co","inc","ltd","us","uk","eu","id","url","api","ml","ai"}
        for t in texts:
            for w in t.split():
                w_clean = w.strip('.,;:!?')
                low = w_clean.lower()
                if len(w_clean) >=2 and len(w_clean) <=6 and abbrev_pattern.match(w_clean):
                    self.add_entry(low, "abbrev")
                elif low in common_abbrevs or (w_clean.endswith('.') and len(w_clean) <=6):
                    self.add_entry(low, "abbrev")
        # Persist mappings
        self._save_index()
        # Optional BitDrop compression of codebook files
        self._compress_codebook_with_bitdrop()

    def _compress_codebook_with_bitdrop(self):
        try:
            import bitdrop_v2 as _bd2
            import io
            for layer_name, cb in self.layers.items():
                # Compress code_map.npy if exists
                map_path = Path(cb.dir) / "code_map.npy"
                if map_path.exists():
                    data = map_path.read_bytes()
                    # Use bitdrop adaptive compression
                    comp = _bd2.compress_adaptive_py(data)
                    out_path = map_path.with_suffix('.npy.bd2')
                    out_path.write_bytes(comp)
                    # Keep original for now
            # Compress index json
            idx_path = self.dir / "text_ref.json"
            if idx_path.exists():
                data = idx_path.read_bytes()
                comp = _bd2.compress_adaptive_py(data)
                out_path = idx_path.with_suffix('.json.bd2')
                out_path.write_bytes(comp)
        except Exception:
            # BitDrop optional
            pass

    def _save_index(self):
        idx_path = self.dir / "text_ref.json"
        # JSON can't store tuple keys, store as "layer:code"
        serial = {k: f"{v[0]}:{v[1]}" for k,v in self.text_to_ref.items()}
        with open(idx_path, 'w') as f:
            json.dump(serial, f)

    def _load_index(self):
        idx_path = self.dir / "text_ref.json"
        if idx_path.exists():
            with open(idx_path) as f:
                serial = json.load(f)
            self.text_to_ref = {}
            self.ref_to_text = {}
            for k,v in serial.items():
                layer, code = v.split(':')
                code = int(code)
                self.text_to_ref[k] = (layer, code)
                self.ref_to_text[(layer,code)] = k

    def encode(self, text: str, chunk_size: int = 256, overlap: int = 32) -> List[Tuple[str,int]]:
        """
        Greedy longest-match encode with layered sliding window disambiguation.
        Returns list of (layer, code_id) references.
        Optimizes for token count and layer priority.
        Supports chunked sliding window for large texts.
        """
        # Cache lookup for full text if small
        if chunk_size >= 1_000_000:
            if text in self._encode_cache:
                return self._encode_cache[text]
        words = text.split()
        if len(words) <= chunk_size:
            # Process whole text
            return self._encode_chunk(words)
        # Sliding window over words
        out = []
        start = 0
        prev_end = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk_words = words[start:end]
            chunk_refs = self._encode_chunk(chunk_words)
            # Avoid duplicate encoding of overlap region
            if start > 0:
                # Drop refs that overlap with previous chunk's tail
                overlap_words = words[max(0, start-overlap):start]
                # Simple heuristic: trim first few refs to avoid duplication
                # Keep at least 0 refs
                trim = min(len(chunk_refs), max(1, overlap//4))
                chunk_refs = chunk_refs[trim:]
            out.extend(chunk_refs)
            if end >= len(words):
                break
            start = end - overlap if overlap > 0 else end
            if start <= prev_end:
                start = end
            prev_end = end
        # Cache
        if len(self._encode_cache) < self._encode_cache_max and chunk_size >= 1_000_000:
            self._encode_cache[text] = out
        return out

    def _encode_chunk(self, words):
        # Cache lookup for chunk text
        chunk_text = ' '.join(words)
        if chunk_text in self._encode_cache:
            return self._encode_cache[chunk_text]
        # Greedy longest-match with layered sliding window disambiguation
        i = 0
        out = []
        # Layer priority for disambiguation: higher is better
        layer_priority = {"technical": 4, "phrases": 3, "abbrev": 2.5, "common": 2}
        CONTEXT_WINDOW = 2
        # Sliding window context memory for disambiguation
        recent_phrases = []
        while i < len(words):
            best_match = None
            best_score = -1
            best_len = 0
            max_n = min(4, len(words) - i)
            # Collect candidates
            candidates = []
            for n in range(max_n, 0, -1):
                phrase = ' '.join(words[i:i+n])
                if phrase in self.text_to_ref:
                    layer, code = self.text_to_ref[phrase]
                    candidates.append((phrase, layer, code, n))
            # Score candidates with layer priority + length + context coherence
            for phrase, layer, code, n in candidates:
                score = n * 10 + layer_priority.get(layer, 0)
                # Context coherence boost: if phrase continues recent pattern
                if recent_phrases:
                    # Simple heuristic: prefer phrases that share words with recent context
                    recent_text = ' '.join(recent_phrases[-CONTEXT_WINDOW:])
                    if recent_text and phrase.split()[0] in recent_text:
                        score += 1
                # Prefer longer matches when scores tie
                if score > best_score:
                    best_score = score
                    best_match = (layer, code)
                    best_len = n
                    best_phrase = phrase
            if best_match:
                layer, code = best_match
                out.append((layer, code))
                recent_phrases.append(best_phrase)
                # Keep recent_phrases bounded
                if len(recent_phrases) > 8:
                    recent_phrases.pop(0)
                i += best_len
            else:
                # No codebook match, skip for BPE fallback
                i += 1
        # Cache result
        if len(self._encode_cache) < self._encode_cache_max:
            self._encode_cache[chunk_text] = out
        return out

    def decode(self, refs: List[Tuple[str,int]]) -> str:
        texts = []
        for layer, code in refs:
            txt = self.ref_to_text.get((layer,code))
            if txt is None:
                # try to recover from layer codebook
                cb = self.layers[layer]
                # reverse lookup code -> value
                # We stored text_to_ref, so this should not happen
                txt = f"<unk:{layer}:{code}>"
            texts.append(txt)
        return ' '.join(texts)

    def token_count(self, text: str) -> int:
        return len(self.encode(text))

# Example usage
if __name__ == "__main__":
    cb = HierarchicalTokenCodebook(dir_path="runtime/token_codebook")
    sample = [
        "machine learning is great",
        "thank you very much for your help",
        "AI stands for artificial intelligence",
        "TYVM",
    ]
    cb.build_from_corpus(sample, min_count=1)
    s = "machine learning is great thank you very much"
    refs = cb.encode(s)
    print("Refs:", refs)
    print("Decoded:", cb.decode(refs))
    print("Token count:", cb.token_count(s))
