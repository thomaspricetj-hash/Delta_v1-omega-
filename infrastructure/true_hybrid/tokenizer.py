"""
Tokenizer stub for True Hybrid Architecture
"""
class Tokenizer:
    def __init__(self, vocab_size=512):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.id2tok = {}
        # simple word-level vocab built on the fly
    def encode(self, text: str):
        tokens = text.lower().split()
        ids = []
        for t in tokens:
            if t not in self.vocab:
                idx = len(self.vocab) % self.vocab_size
                self.vocab[t] = idx
                self.id2tok[idx] = t
            ids.append(self.vocab[t])
        return ids
    def decode(self, ids):
        return " ".join([self.id2tok.get(i, "<unk>") for i in ids])
