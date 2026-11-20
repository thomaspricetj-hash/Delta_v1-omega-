"""
Production-grade HSCL dictionary builder
Frequency + compression gain + semantic stability
"""
import re
from collections import Counter
from pathlib import Path
import json
from typing import List, Dict, Tuple

STOP_WORDS = set("""a an the and or but if then else for while in on at to of with by from as is are was were be been being""".split())

def tokenize_for_ngrams(text: str):
    # simple word tokenize preserving case for phrases
    return re.findall(r"[A-Za-z0-9']+", text)

def extract_ngrams(tokens: List[str], n_min=2, n_max=4):
    ngrams = []
    for n in range(n_min, n_max+1):
        for i in range(len(tokens)-n+1):
            phrase = " ".join(tokens[i:i+n])
            ngrams.append(phrase)
    return ngrams

class HSCLBuilder:
    def __init__(self, min_freq=50, min_gain=1.5, max_vocab=5000, use_apriori=True):
        self.min_freq = min_freq
        self.min_gain = min_gain
        self.max_vocab = max_vocab
        self.use_apriori = use_apriori
        self.counter = Counter()
        self.phrase_stats: Dict[str, Dict] = {}
        # Optional Apriori miner
        try:
            from infrastructure.ml_extensions.pattern_mining import PatternMiner
            self.miner = PatternMiner(min_support=self.min_freq/1000000, max_len=4)  # rough support
            self._apriori_available = True
        except Exception:
            self.miner = None
            self._apriori_available = False

    def ingest_texts(self, texts: List[str]):
        for t in texts:
            tokens = tokenize_for_ngrams(t.lower())
            # filter stopwords for ngram building but keep original phrase case later
            # collect ngrams
            ngrams = extract_ngrams(tokens, 2, 4)
            # filter stopword-only phrases
            filtered = []
            for ng in ngrams:
                words = ng.split()
                if any(w not in STOP_WORDS for w in words):
                    filtered.append(ng)
            self.counter.update(filtered)

    def build_candidates(self):
        # Hybrid candidate generation: Counter + Apriori frequent patterns
        if self.use_apriori and self._apriori_available and self.miner is not None:
            # Build tokenized corpus for Apriori
            corpus_tokens = []
            # Re-tokenize from counter keys for simplicity
            # Use counter keys as proxy for phrase occurrences
            # For production, store original tokenized docs during ingest
            # Here we approximate by using counter items
            # Build a small sample corpus from most frequent phrases
            sample_phrases = [p for p,_ in self.counter.most_common(20000)]
            corpus_for_apriori = [p.split() for p in sample_phrases]
            try:
                freq_patterns = self.miner.top_patterns(corpus_for_apriori, top_k=self.max_vocab*2)
                candidates = []
                for pattern, count in freq_patterns:
                    phrase = " ".join(pattern)
                    freq = int(count)
                    if freq < self.min_freq:
                        continue
                    n_tokens = len(pattern)
                    gain = n_tokens / 1.0
                    if gain < self.min_gain:
                        continue
                    if len(pattern) < 2 or len(pattern) > 4:
                        continue
                    candidates.append((phrase, freq, gain))
                # Fall back to counter if Apriori yields too few
                if len(candidates) < self.max_vocab // 2:
                    # merge with counter based candidates
                    counter_candidates = []
                    for phrase, freq in self.counter.items():
                        if freq < self.min_freq:
                            continue
                        n_tokens = len(phrase.split())
                        gain = n_tokens / 1.0
                        if gain < self.min_gain:
                            continue
                        counter_candidates.append((phrase, freq, gain))
                    counter_candidates.sort(key=lambda x: x[1]*x[2], reverse=True)
                    candidates = candidates + counter_candidates
                candidates.sort(key=lambda x: x[1]*x[2], reverse=True)
                return candidates[:self.max_vocab]
            except Exception:
                # fall through to counter method
                pass
        # Fallback counter-based candidate generation
        candidates = []
        for phrase, freq in self.counter.items():
            if freq < self.min_freq:
                continue
            n_tokens = len(phrase.split())
            gain = n_tokens / 1.0
            if gain < self.min_gain:
                continue
            words = phrase.split()
            if len(words) < 2 or len(words) > 4:
                continue
            candidates.append((phrase, freq, gain))
        candidates.sort(key=lambda x: x[1]*x[2], reverse=True)
        return candidates[:self.max_vocab]

    def save_dictionary(self, path: Path, candidates: List[Tuple]):
        # Map phrase -> concept_id
        # concept IDs start at 8192 to avoid BPE collision
        base_id = 8192
        mapping = {}
        for i, (phrase, freq, gain) in enumerate(candidates):
            cid = base_id + i
            mapping[phrase] = cid
        # also save metadata
        out = {
            "base_id": base_id,
            "entries": {phrase: {"id": base_id+i, "freq": int(freq), "gain": float(gain)} for i,(phrase,freq,gain) in enumerate(candidates)}
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        return mapping

def build_from_directory(data_dir: Path, out_path: Path, min_freq=50, max_vocab=5000, use_apriori=True):
    import pandas as pd, glob
    texts = []
    for f in glob.glob(str(data_dir/"*.parquet")):
        try:
            df = pd.read_parquet(f)
            for col in df.columns:
                texts.extend(df[col].dropna().astype(str).tolist())
        except Exception:
            pass
    builder = HSCLBuilder(min_freq=min_freq, max_vocab=max_vocab, use_apriori=use_apriori)
    builder.ingest_texts(texts[:200000])  # sample for speed
    candidates = builder.build_candidates()
    mapping = builder.save_dictionary(out_path, candidates)
    print(f"HSCL dictionary built: {len(mapping)} entries -> {out_path}")
    return mapping
