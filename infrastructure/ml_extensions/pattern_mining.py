"""
Production-grade frequent pattern mining for HSCL.
Apriori implementation with efficient pruning.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from typing import List, Set, Tuple, Dict
import itertools

class PatternMiner:
    """
    Apriori for phrase discovery in tokenized corpora.
    Works on sequences of tokens/phrases.
    """
    def __init__(self, min_support: float = 0.001, max_len: int = 4):
        self.min_support = min_support
        self.max_len = max_len

    def _transactions(self, corpus: List[List[str]]) -> List[Set[str]]:
        return [set(doc) for doc in corpus]

    def apriori(self, corpus: List[List[str]]) -> Dict[Tuple[str, ...], int]:
        transactions = self._transactions(corpus)
        n_trans = len(transactions)
        min_count = max(1, int(self.min_support * n_trans))

        # 1-itemsets
        item_counts = Counter()
        for t in transactions:
            for item in t:
                item_counts[item] += 1
        freq_itemsets = {frozenset([i]): c for i, c in item_counts.items() if c >= min_count}

        all_freq = dict()
        k = 2
        current = freq_itemsets
        while current and k <= self.max_len:
            # generate candidates
            candidates = {}
            items = list(current.keys())
            for i in range(len(items)):
                for j in range(i+1, len(items)):
                    union = items[i] | items[j]
                    if len(union) == k:
                        # prune
                        subsets = [frozenset(s) for s in itertools.combinations(union, k-1)]
                        if all(s in current for s in subsets):
                            candidates[union] = 0
            # count
            for t in transactions:
                for cand in candidates:
                    if cand.issubset(t):
                        candidates[cand] += 1
            # filter
            current = {c: cnt for c, cnt in candidates.items() if cnt >= min_count}
            for c, cnt in current.items():
                all_freq[tuple(sorted(c))] = cnt
            k += 1

        # include 1-itemsets
        for kset, cnt in freq_itemsets.items():
            all_freq[tuple(sorted(kset))] = cnt
        return all_freq

    def top_patterns(self, corpus: List[List[str]], top_k: int = 1000) -> List[Tuple[Tuple[str, ...], int]]:
        freq = self.apriori(corpus)
        sorted_items = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:top_k]
