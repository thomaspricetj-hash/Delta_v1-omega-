"""
Global Bloom filter + skimming cache shared across Pattern / Helper / Micro trees
"""
import numpy as np
import hashlib

class BloomFilter:
    def __init__(self, size=1<<22, hash_count=4):
        self.size = size
        self.hash_count = hash_count
        self.bit_array = np.zeros(size, dtype=np.bool_)
    def _hashes(self, key):
        for i in range(self.hash_count):
            h = hashlib.md5((str(key)+str(i)).encode()).hexdigest()
            yield int(h,16) % self.size
    def add(self, key):
        for h in self._hashes(key):
            self.bit_array[h] = True
    def __contains__(self, key):
        return all(self.bit_array[h] for h in self._hashes(key))

# Global shared instances
_global_bloom = BloomFilter(size=1<<22)
_global_cache = {}

def get_global_bloom():
    return _global_bloom

def get_global_cache():
    return _global_cache
