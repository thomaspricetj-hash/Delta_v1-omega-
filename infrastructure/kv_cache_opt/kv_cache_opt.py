"""
KV Cache Optimization + Hierarchical Memory
Reuse Keys/Values, hierarchical summarization
"""
from typing import Dict, List
import numpy as np

class KVCacheOptimizer:
    def __init__(self):
        self.cache = {}
        self.hierarchical_summaries = {}

    def store_kv(self, layer_id: int, key: np.ndarray, value: np.ndarray):
        self.cache[(layer_id, id(key))] = (key, value)

    def reuse_kv(self, layer_id: int, key_id):
        return self.cache.get((layer_id, key_id))

    def hierarchical_memory_lookup(self, query: str, levels: List[str] = None):
        # Book -> Chapters -> Paragraphs -> Sentences
        if levels is None:
            levels = ['book','chapter','paragraph','sentence']
        # Return summary at top level first, drill down only when needed
        for level in levels:
            summary = self.hierarchical_summaries.get(level)
            if summary:
                return summary
        return None

    def update_hierarchical_summary(self, level: str, summary):
        self.hierarchical_summaries[level] = summary

# Singleton
_kv_opt = KVCacheOptimizer()

def store_kv(layer_id, key, value):
    _kv_opt.store_kv(layer_id, key, value)

def reuse_kv(layer_id, key_id):
    return _kv_opt.reuse_kv(layer_id, key_id)

def hierarchical_lookup(query):
    return _kv_opt.hierarchical_memory_lookup(query)

def update_summary(level, summary):
    _kv_opt.update_hierarchical_summary(level, summary)
