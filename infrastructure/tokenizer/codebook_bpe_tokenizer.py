"""
Codebook-aware tokenizer wrapper
Uses HierarchicalTokenCodebook for phrase compression, falls back to BPE.
Optimizes for fewer tokens.
"""
from pathlib import Path
import hashlib
from infrastructure.tokenizer.bpe_tokenizer import get_bpe_tokenizer
from infrastructure.true_hybrid.token_codebook_tokenizer import HierarchicalTokenCodebook

class BloomFilter:
    def __init__(self, size=1<<20, hashes=4):
        self.size = size
        self.hashes = hashes
        self.bits = bytearray(size // 8)
    def _hashes(self, key):
        data = key.encode('utf-8')
        for i in range(self.hashes):
            h = hashlib.md5(data + str(i).encode()).hexdigest()
            yield int(h, 16) % self.size
    def add(self, key):
        for h in self._hashes(key):
            self.bits[h // 8] |= 1 << (h % 8)
    def __contains__(self, key):
        return all((self.bits[h // 8] >> (h % 8)) & 1 for h in self._hashes(key))

class CodebookBPETokenizer:
    def __init__(self, codebook_path: str = "runtime/token_codebook", capacity_per_node: int = 1024):
        self.bpe = get_bpe_tokenizer()
        self.codebook = HierarchicalTokenCodebook(dir_path=codebook_path, capacity_per_node=capacity_per_node)
        self._codebook_vocab_offset = None
        self._ref_to_id = {}
        self._id_to_ref = {}
        self._phrase_bloom = BloomFilter(size=1<<22, hashes=4)
        # Dictionary layer for fast word recall
        self._word_dict = {}
        self._dict_to_id = {}
        self._dict_offset = None
        self._load_word_dictionary()
        self._ensure_codebook_vocab()
        # Freeze codebook for stable encoding/decoding during training/inference
        self.codebook.frozen = True

    def _load_word_dictionary(self):
        import json, os
        dict_path = Path("runtime/token_dictionary.json")
        if dict_path.exists():
            try:
                with open(dict_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self._word_dict = data.get('word_to_id', {})
                # Build reverse map
                self._dict_to_id = {v:k for k,v in self._word_dict.items()}
            except Exception:
                self._word_dict = {}
                self._dict_to_id = {}
        else:
            self._word_dict = {}
            self._dict_to_id = {}
        # Dictionary offset sits after BPE vocab
        self._dict_offset = len(self.bpe.vocab)

    def _ensure_codebook_vocab(self):
        bpe_vocab_size = len(self.bpe.vocab)
        if self._codebook_vocab_offset is None:
            self._codebook_vocab_offset = bpe_vocab_size
        # Load persisted index first so mappings are built from disk
        self.codebook._load_index()
        # Rebuild id mappings and bloom filter
        self._ref_to_id.clear()
        self._id_to_ref.clear()
        # Reset bloom
        self._phrase_bloom = BloomFilter(size=1<<22, hashes=4)
        for text, (layer, code) in self.codebook.text_to_ref.items():
            ref = (layer, code)
            if ref not in self._ref_to_id:
                new_id = self._codebook_vocab_offset + len(self._ref_to_id)
                self._ref_to_id[ref] = new_id
                self._id_to_ref[new_id] = ref
            # Populate bloom for fast negative checks
            self._phrase_bloom.add(text)

    def _encode_words(self, words):
        # Internal word list encoder, no chunking
        ids = []
        i = 0
        text_to_ref = self.codebook.text_to_ref
        bpe_buffer = []
        # Safety guard against pathological loops
        max_iter = len(words) * 2 + 1000
        iter_count = 0
        # Guard against extremely long single tokens
        max_word_len = 200
        def flush_bpe_buffer():
            nonlocal bpe_buffer
            if bpe_buffer:
                bpe_text = ' '.join(bpe_buffer)
                try:
                    bpe_ids = self.bpe.encode(bpe_text)
                    ids.extend([tok for tok in bpe_ids if 0 <= tok < len(self.bpe.vocab)])
                except Exception:
                    for w in bpe_buffer:
                        try:
                            bpe_ids = self.bpe.encode(w)
                            ids.extend([tok for tok in bpe_ids if 0 <= tok < len(self.bpe.vocab)])
                        except Exception:
                            ids.extend([self.bpe.inverse_vocab.get(ch, 0) for ch in w])
                bpe_buffer = []
        while i < len(words) and iter_count < max_iter:
            iter_count += 1
            # Fast dictionary lookup for single words
            word = words[i]
            # Truncate extremely long tokens
            if len(word) > max_word_len:
                word = word[:max_word_len]
            if word in self._word_dict:
                flush_bpe_buffer()
                ids.append(self._word_dict[word])
                i += 1
                continue
            matched = None
            matched_len = 0
            for n in (3, 2, 1):
                if i + n > len(words):
                    continue
                phrase = ' '.join(words[i:i+n])
                # Fast negative check with bloom filter
                if phrase not in self._phrase_bloom:
                    continue
                if phrase in text_to_ref:
                    matched = phrase
                    matched_len = n
                    break
            if matched:
                flush_bpe_buffer()
                layer, code = text_to_ref[matched]
                ref = (layer, code)
                if ref not in self._ref_to_id:
                    new_id = self._codebook_vocab_offset + len(self._ref_to_id)
                    self._ref_to_id[ref] = new_id
                    self._id_to_ref[new_id] = ref
                ids.append(self._ref_to_id[ref])
                i += matched_len
            else:
                bpe_buffer.append(words[i])
                i += 1
        flush_bpe_buffer()
        return ids

    def encode(self, text: str, chunk_size: int = 256, overlap: int = 32, max_words: int = 2048, max_chars: int = 200000):
        # Hybrid codebook + BPE encoding with chunking / sliding window
        if not self._ref_to_id:
            self._ensure_codebook_vocab()
        # Guard against pathological very long texts
        if len(text) > max_chars:
            text = text[:max_chars]
        words = text.split()
        # Guard against pathological very long texts
        if len(words) > max_words:
            words = words[:max_words]
        if len(words) <= chunk_size:
            return self._encode_words(words)
        ids = []
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk_words = words[start:end]
            chunk_ids = self._encode_words(chunk_words)
            ids.extend(chunk_ids)
            # Advance start with overlap, ensure progress
            next_start = end - overlap if overlap > 0 else end
            if next_start <= start:
                # Prevent infinite loop on final short chunk
                next_start = end
            start = next_start
        return ids

    def decode(self, token_ids):
        bpe_max = len(self.bpe.vocab)
        dict_size = len(self._word_dict)
        dict_start = bpe_max
        dict_end = dict_start + dict_size
        codebook_start = self._codebook_vocab_offset
        out_parts = []
        bpe_buf = []
        def flush_bpe():
            if bpe_buf:
                out_parts.append(self.bpe.decode(bpe_buf))
                bpe_buf.clear()
        for tid in token_ids:
            if tid < bpe_max:
                bpe_buf.append(tid)
            elif dict_size > 0 and dict_start <= tid < dict_end:
                flush_bpe()
                word = self._dict_to_id.get(tid, "")
                out_parts.append(word)
            else:
                flush_bpe()
                ref = self._id_to_ref.get(tid)
                if ref:
                    layer, code = ref
                    txt = self.codebook.ref_to_text.get(ref)
                    if txt:
                        out_parts.append(txt)
                    else:
                        out_parts.append(f"<{layer}:{code}>")
                else:
                    out_parts.append("")
        flush_bpe()
        # Join parts with space; BPE runs are already flushed together
        return " ".join(out_parts)

    def build_codebook(self, texts, min_count=5):
        self.codebook.build_from_corpus(texts, min_count=min_count)
        self._ref_to_id.clear()
        self._id_to_ref.clear()
        self._ensure_codebook_vocab()

_codebook_tokenizer = None

def get_codebook_tokenizer(codebook_path="runtime/token_codebook"):
    global _codebook_tokenizer
    if _codebook_tokenizer is None:
        _codebook_tokenizer = CodebookBPETokenizer(codebook_path=codebook_path)
    return _codebook_tokenizer
