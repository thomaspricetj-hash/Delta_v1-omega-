"""
Pattern Classifier for BitDrop Engine
Maps raw bit patterns to abstract rule-based classes before attention/reasoning.
"""
from collections import defaultdict
from typing import Dict, List, Tuple

class PatternClassifier:
    def __init__(self):
        # Example rule-based classes
        self.class_map = {
            '1100': 'A',
            '1011': 'B',
            '1001': 'C',
        }
        self.class_counts = defaultdict(int)
        self.pattern_to_class_cache = {}

    def classify_pattern(self, bits: str) -> str:
        if bits in self.pattern_to_class_cache:
            return self.pattern_to_class_cache[bits]
        cls = self.class_map.get(bits, 'UNKNOWN')
        self.pattern_to_class_cache[bits] = cls
        self.class_counts[cls] += 1
        return cls

    def classify_bytes(self, data: bytes) -> List[Tuple[str,str]]:
        # Simple 4-bit window classifier
        out = []
        window = ''
        for b in data:
            bits = format(b, '08b')
            for bit in bits:
                window += bit
                if len(window) >= 4:
                    pat = window[-4:]
                    cls = self.classify_pattern(pat)
                    out.append((pat, cls))
        return out

    def merge_similar_classes(self, threshold: int = 5000):
        # Merge classes with similar counts
        merges = []
        classes = list(self.class_counts.items())
        for i in range(len(classes)):
            for j in range(i+1, len(classes)):
                c1, cnt1 = classes[i]
                c2, cnt2 = classes[j]
                if abs(cnt1 - cnt2) < threshold:
                    merges.append((c1, c2))
        return merges

    def class_level_multiply(self, class_a: str, class_b: str) -> str:
        # Abstract multiplication at class level
        return f"{class_a}*{class_b}"

    def get_stats(self):
        return dict(self.class_counts)

# Singleton
_classifier = PatternClassifier()

def classify(data: bytes):
    return _classifier.classify_bytes(data)

def get_class_stats():
    return _classifier.get_stats()

def merge_classes(threshold=5000):
    return _classifier.merge_similar_classes(threshold)
