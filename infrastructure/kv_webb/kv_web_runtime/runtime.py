"""
KV-Webb Runtime – drift, pruning, dynamic webbing, semantic clustering
"""
from pathlib import Path
from kv_web_core.kv_web import KvWeb
import re

class KvWebRuntime:
    def __init__(self, storage_path: Path = None):
        if storage_path is None:
            storage_path = Path("C:/Users/thoma/.lmstudio/apps/bionic/projects/d49037d8-47f4-5808-9028-c707de117f8f/workspace/TRANSLATION_OS_V1/runtime/logs/kv_webb.json")
        self.web = KvWeb(storage_path)

    def tokenize(self, text: str):
        return re.findall(r'\b\w+\b', text.lower())

    def write_interaction(self, query: str, answer: str, reward: float = 1.0):
        q_tokens = self.tokenize(query)
        a_tokens = self.tokenize(answer)
        q_node = self.web.create_node(q_tokens, label="query", score=reward)
        a_node = self.web.create_node(a_tokens, label="answer", score=reward)
        self.web.add_edge(q_node.id, a_node.id, weight=reward, kind="Semantic")
        self.web.reinforce([q_node.id, a_node.id], reward)
        self.web._save()

    def retrieve(self, query: str, top_k: int = 20):
        tokens = self.tokenize(query)
        nodes = self.web.retrieve_region(tokens, top_k)
        return nodes

    def maintenance(self):
        self.web.apply_drift(decay=0.001)
        pruned = self.web.prune(score_threshold=0.01)
        return {"pruned": pruned}
