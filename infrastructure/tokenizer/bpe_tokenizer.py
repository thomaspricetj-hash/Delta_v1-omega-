"""
BPE Tokenizer - educational implementation
Based on provided BPETokenizerSimple
"""
from collections import Counter, deque
from functools import lru_cache
import json
import re

class BPETokenizerSimple:
    def __init__(self):
        self.vocab = {}
        self.inverse_vocab = {}
        self.bpe_merges = {}
        self.bpe_ranks = {}

    def train(self, text, vocab_size, allowed_special={"<|endoftext|>"}, incremental=False):
        # Process text for BPE space handling
        processed_chars = []
        for i, char in enumerate(text):
            if char == " " and i != 0:
                processed_chars.append("Ġ")
            if char != " ":
                processed_chars.append(char)
        processed_text = "".join(processed_chars)

        if not incremental:
            unique_chars = [chr(i) for i in range(256)]
            unique_chars.extend(char for char in sorted(set(processed_text)) if char not in unique_chars)
            if "Ġ" not in unique_chars:
                unique_chars.append("Ġ")

            self.vocab = {i: char for i, char in enumerate(unique_chars)}
            self.inverse_vocab = {char: i for i, char in self.vocab.items()}
        else:
            # For incremental training, ensure vocab exists
            if not self.vocab:
                raise ValueError("Vocab not initialized for incremental training")

        if allowed_special:
            for token in allowed_special:
                if token not in self.inverse_vocab:
                    new_id = len(self.vocab)
                    self.vocab[new_id] = token
                    self.inverse_vocab[token] = new_id

        token_ids = [self.inverse_vocab[char] for char in processed_text]

        for new_id in range(len(self.vocab), vocab_size):
            pair_id = self.find_freq_pair(token_ids, mode="most")
            if pair_id is None:
                break
            token_ids = self.replace_pair(token_ids, pair_id, new_id)
            self.bpe_merges[pair_id] = new_id

        for (p0, p1), new_id in self.bpe_merges.items():
            merged_token = self.vocab[p0] + self.vocab[p1]
            self.vocab[new_id] = merged_token
            self.inverse_vocab[merged_token] = new_id

    def encode(self, text, allowed_special=None):
        token_ids = []
        if allowed_special is not None and len(allowed_special) > 0:
            special_pattern = "(" + "|".join(re.escape(tok) for tok in sorted(allowed_special, key=len, reverse=True)) + ")"
            last_index = 0
            for match in re.finditer(special_pattern, text):
                prefix = text[last_index:match.start()]
                token_ids.extend(self.encode(prefix, allowed_special=None))
                special_token = match.group(0)
                if special_token in self.inverse_vocab:
                    token_ids.append(self.inverse_vocab[special_token])
                else:
                    raise ValueError(f"Special token {special_token} not found")
                last_index = match.end()
            text = text[last_index:]

        tokens = []
        lines = text.split("\n")
        for i, line in enumerate(lines):
            if i > 0:
                tokens.append("\n")
            words = line.split()
            for j, word in enumerate(words):
                if j == 0 and i > 0:
                    tokens.append("Ġ" + word)
                elif j == 0:
                    tokens.append(word)
                else:
                    tokens.append("Ġ" + word)

        for token in tokens:
            if token in self.inverse_vocab:
                token_ids.append(self.inverse_vocab[token])
            else:
                token_ids.extend(self.tokenize_with_bpe(token))
        return token_ids

    def tokenize_with_bpe(self, token):
        # Fallback unknown characters to space to avoid crashes on non-ASCII input
        space_id = self.inverse_vocab.get(' ', 0)
        token_ids = [self.inverse_vocab.get(char, space_id) for char in token]
        if not self.bpe_ranks:
            can_merge = True
            while can_merge and len(token_ids) > 1:
                can_merge = False
                new_tokens = []
                i = 0
                while i < len(token_ids)-1:
                    pair = (token_ids[i], token_ids[i+1])
                    if pair in self.bpe_merges:
                        merged_token_id = self.bpe_merges[pair]
                        new_tokens.append(merged_token_id)
                        i += 2
                        can_merge = True
                    else:
                        new_tokens.append(token_ids[i])
                        i += 1
                if i < len(token_ids):
                    new_tokens.append(token_ids[i])
                token_ids = new_tokens
            return token_ids
        symbols = [self.vocab[id_num] for id_num in token_ids]
        while True:
            pairs = set(zip(symbols, symbols[1:]))
            if not pairs:
                break
            min_rank = float("inf")
            bigram = None
            for p in pairs:
                r = self.bpe_ranks.get(p, float("inf"))
                if r < min_rank:
                    min_rank = r
                    bigram = p
            if bigram is None or bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_symbols = []
            i = 0
            while i < len(symbols):
                if i < len(symbols)-1 and symbols[i]==first and symbols[i+1]==second:
                    new_symbols.append(first+second)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols
            if len(symbols)==1:
                break
        return [self.inverse_vocab[sym] for sym in symbols]

    def decode(self, token_ids):
        out = ""
        for token_id in token_ids:
            token = self.vocab.get(token_id, "")
            if token == "\n":
                out += " "
            elif token.startswith("Ġ"):
                out += " " + token[1:]
            else:
                out += token
        return out

    @staticmethod
    def find_freq_pair(token_ids, mode="most"):
        pairs = Counter(zip(token_ids, token_ids[1:]))
        if not pairs:
            return None
        if mode=="most":
            return max(pairs.items(), key=lambda x: x[1])[0]
        else:
            return min(pairs.items(), key=lambda x: x[1])[0]

    @staticmethod
    def replace_pair(token_ids, pair_id, new_id):
        dq = deque(token_ids)
        replaced = []
        while dq:
            cur = dq.popleft()
            if dq and (cur, dq[0]) == pair_id:
                replaced.append(new_id)
                dq.popleft()
            else:
                replaced.append(cur)
        return replaced

# Singleton
_bpe_tokenizer = BPETokenizerSimple()

def _load_tokenizer_from_disk(tok):
    import json
    from pathlib import Path
    base = Path(__file__).resolve().parents[2]
    vocab_path = base / "runtime" / "tokenizer" / "vocab.json"
    merges_path = base / "runtime" / "tokenizer" / "bpe_merges.json"
    try:
        if vocab_path.exists():
            raw_vocab = json.loads(vocab_path.read_text())
            # JSON keys are strings, convert back to int
            tok.vocab = {int(k): v for k, v in raw_vocab.items()}
            tok.inverse_vocab = {v:k for k,v in tok.vocab.items()}
        if merges_path.exists():
            raw = json.loads(merges_path.read_text())
            # merges saved as string keys "a:b": new_id
            merges = {}
            for k,v in raw.items():
                try:
                    a_str, b_str = k.split(":",1)
                    a = int(a_str); b = int(b_str)
                    merges[(a,b)] = v
                except Exception:
                    pass
            tok.bpe_merges = merges
    except Exception:
        pass

def get_bpe_tokenizer():
    if not _bpe_tokenizer.vocab:
        _load_tokenizer_from_disk(_bpe_tokenizer)
    return _bpe_tokenizer
