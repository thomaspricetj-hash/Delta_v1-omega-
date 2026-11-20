"""
Codebook-backed PyTorch Dataset for WeaveFormer training.
Encodes on the fly with the CodebookBPETokenizer and returns token ids.
"""
from torch.utils.data import Dataset, DataLoader
from infrastructure.tokenizer.codebook_bpe_tokenizer import get_codebook_tokenizer

class CodebookTextDataset(Dataset):
    def __init__(self, texts, tokenizer=None, max_tokens_per_text=8192):
        self.texts = texts
        self.tokenizer = tokenizer or get_codebook_tokenizer()
        self.max_tokens = max_tokens_per_text
        self._cache = {}
        self._cache_max = 10000

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        if idx in self._cache:
            return self._cache[idx]
        if idx < 5:
            print(f"[Dataset] Encoding idx {idx} len={len(self.texts[idx])}")
        text = self.texts[idx]
        # Simple clean
        text = ''.join(ch if ord(ch) < 128 else ' ' for ch in text)
        # Use layered sliding-window encode for large texts
        ids = self.tokenizer.encode(text, chunk_size=512, overlap=64)
        if len(ids) > self.max_tokens:
            ids = ids[:self.max_tokens]
        # Cache with simple LRU eviction
        if len(self._cache) >= self._cache_max:
            # pop arbitrary item
            self._cache.pop(next(iter(self._cache)))
        self._cache[idx] = ids
        return ids

def collate_codebook_batch(batch, max_seq_len=128):
    # batch is list of list[int]
    # Pad to max_seq_len and create input/target pairs
    import torch.nn.utils.rnn as rnn_utils
    import torch
    # Truncate each sequence to max_seq_len
    seqs = [s[:max_seq_len] for s in batch]
    # Create input/target by shifting
    inputs = []
    targets = []
    for s in seqs:
        if len(s) < 2:
            continue
        inputs.append(s[:-1])
        targets.append(s[1:])
    if not inputs:
        # Debug: print lengths
        lens = [len(s) for s in seqs]
        print(f"[Collate] Empty batch, seq lens {lens}")
        return None
    input_padded = rnn_utils.pad_sequence([torch.tensor(x, dtype=torch.long) for x in inputs], batch_first=True, padding_value=0)
    target_padded = rnn_utils.pad_sequence([torch.tensor(x, dtype=torch.long) for x in targets], batch_first=True, padding_value=0)
    return input_padded, target_padded
