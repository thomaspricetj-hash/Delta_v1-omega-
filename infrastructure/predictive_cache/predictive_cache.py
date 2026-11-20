"""
Predictive Memory Cache
Predictive Cache + Active Memory + Long-Term Memory -> Merge -> Read Head
"""
import re
from collections import defaultdict, deque
from pathlib import Path

class PredictiveMemoryCache:
    def __init__(self, history_window=50, top_k=20):
        self.history_window = history_window
        self.top_k = top_k
        self.history = deque(maxlen=history_window)
        self.pattern_counts = defaultdict(int)
        self.success_map = defaultdict(float)
        self.cooccurrence = defaultdict(lambda: defaultdict(int))

    def canonicalize(self, text):
        t = text.lower()
        t = re.sub(r'[^\w\s]', ' ', t)
        t = re.sub(r'\s+', ' ', t).strip()
        return t

    def update_history(self, query, response, success=True):
        q_canon = self.canonicalize(query)
        self.history.append(q_canon)
        words = set(q_canon.split())
        for w in words:
            self.pattern_counts[w] += 1
            self.success_map[w] += 1.0 if success else 0.0
        # co-occurrence
        words = list(words)
        for i in range(len(words)):
            for j in range(i+1, len(words)):
                a, b = words[i], words[j]
                self.cooccurrence[a][b] += 1
                self.cooccurrence[b][a] += 1

    def predict_topics(self, query, k=10):
        q_canon = self.canonicalize(query)
        words = set(q_canon.split())
        scores = defaultdict(float)
        # Base on history pattern counts and success
        for w in words:
            for candidate, cnt in self.pattern_counts.items():
                if candidate == w:
                    continue
                co = self.cooccurrence[w].get(candidate, 0)
                base = self.pattern_counts[candidate]
                success = self.success_map[candidate] / max(1, self.pattern_counts[candidate])
                score = base * 0.3 + co * 0.5 + success * 0.2
                scores[candidate] += score
        # Return top k
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        return [w for w,_ in ranked]

    def preload_memories(self, query):
        topics = self.predict_topics(query, k=self.top_k)
        # Simple heuristic: return topics as preload keys
        return topics

    def score_memory(self, mem_key, predictive_score, relevance_score, success_score, recency_score):
        final = (
            predictive_score * 0.3 +
            relevance_score * 0.4 +
            success_score * 0.2 +
            recency_score * 0.1
        )
        return final

# Singleton
_predictive_cache = PredictiveMemoryCache()

def get_predictive_cache():
    return _predictive_cache
