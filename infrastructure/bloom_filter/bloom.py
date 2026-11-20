"""
BitDrop v2 backed Bloom filter for Translation OS
Probabilistic membership test for dedup and fast negative lookups
"""
import hashlib
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE / "scripts"))
from bitdrop_persistence import save_json, load_json_bitdrop

DEFAULT_SIZE = 1 << 24  # 16M bits ~2 MB
DEFAULT_HASHES = 7
STATE_PATH = BASE / "runtime" / "logs" / "bloom_filter_state.json"

def _hashes(item: str, num_hashes, size):
    # Double hashing
    h1 = int(hashlib.sha256(item.encode()).hexdigest(), 16)
    h2 = int(hashlib.sha256((item + "salt").encode()).hexdigest(), 16)
    for i in range(num_hashes):
        yield (h1 + i * h2) % size

class BloomFilter:
    def __init__(self, size=DEFAULT_SIZE, hashes=DEFAULT_HASHES):
        self.size = size
        self.hashes = hashes
        # bitarray as bytearray
        self.bits = bytearray((size + 7) // 8)
        self._load()

    def _load(self):
        data = load_json_bitdrop(STATE_PATH)
        if data and "bits" in data:
            try:
                self.bits = bytearray(data["bits"])
            except Exception:
                pass

    def _save(self):
        save_json(STATE_PATH, {
            "size": self.size,
            "hashes": self.hashes,
            "bits": list(self.bits)
        })

    def _set_bit(self, pos):
        byte_idx = pos // 8
        bit_idx = pos % 8
        self.bits[byte_idx] |= 1 << bit_idx

    def _get_bit(self, pos):
        byte_idx = pos // 8
        bit_idx = pos % 8
        return (self.bits[byte_idx] >> bit_idx) & 1

    def add(self, item: str):
        for pos in _hashes(item, self.hashes, self.size):
            self._set_bit(pos)
        self._save()

    def __contains__(self, item: str):
        return all(self._get_bit(pos) for pos in _hashes(item, self.hashes, self.size))

    def maybe_contains(self, item: str):
        return item in self

# Singleton
_bloom = None
def get_bloom():
    global _bloom
    if _bloom is None:
        _bloom = BloomFilter()
    return _bloom

if __name__ == "__main__":
    bf = get_bloom()
    bf.add("Q42")
    print("Q42 in?", "Q42" in bf)
    print("Q999 in?", "Q999" in bf)
