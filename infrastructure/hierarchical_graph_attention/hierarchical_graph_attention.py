"""
Hierarchical Graph Attention
Grouped tokens → graph edges → hierarchical attention
Reduces O(N²) to O(G² + sum(group²))
"""
from typing import List, Dict
import numpy as np

class HierarchicalGraphAttention:
    def __init__(self, window_size: int = 2, num_groups: int = 4, cluster_threshold: float = 0.7):
        self.window_size = window_size
        self.num_groups = num_groups
        self.cluster_threshold = cluster_threshold

    def group_tokens(self, tokens: List[str]) -> List[List[str]]:
        n = len(tokens)
        if n == 0:
            return []
        group_size = max(1, n // self.num_groups)
        groups = []
        for i in range(self.num_groups):
            start = i * group_size
            end = start + group_size if i < self.num_groups - 1 else n
            groups.append(tokens[start:end])
        return groups

    def build_graph_edges(self, groups: List[List[str]]) -> Dict[int, List[int]]:
        edges = {}
        for i in range(len(groups)):
            neighbors = []
            if i > 0:
                neighbors.append(i-1)
            if i < len(groups)-1:
                neighbors.append(i+1)
            edges[i] = neighbors
        return edges

    def sliding_window_mask(self, seq_len: int) -> np.ndarray:
        mask = np.zeros((seq_len, seq_len), dtype=bool)
        for i in range(seq_len):
            start = max(0, i - self.window_size)
            end = min(seq_len, i + self.window_size + 1)
            mask[i, start:end] = True
        return mask

    def attend_within_group(self, group_embeddings: np.ndarray) -> np.ndarray:
        # Simplified attention within group with sliding window
        mask = self.sliding_window_mask(group_embeddings.shape[0])
        # Placeholder: return masked average
        return group_embeddings * mask

    def attend_between_groups(self, group_reprs: List[np.ndarray]) -> List[np.ndarray]:
        # Inter-group attention via graph edges
        edges = self.build_graph_edges(group_reprs)
        outputs = []
        for i, repr_i in enumerate(group_reprs):
            # aggregate neighbors
            agg = np.mean(repr_i, axis=0, keepdims=True)
            for nb in edges.get(i, []):
                nb_repr = group_reprs[nb]
                agg = (agg + np.mean(nb_repr, axis=0, keepdims=True)) / 2
            outputs.append(agg)
        return outputs

    def cluster_groups(self, groups: List[List[str]]):
        # Simple similarity clustering placeholder
        # Returns clustered groups
        return groups

    def compress_group_context(self, group_emb: np.ndarray):
        # BitDrop v2 compress group context
        try:
            import sys
            sys.path.insert(0, 'infrastructure')
            from bitdrop_v2_integration import compress_data
            raw = group_emb.tobytes()
            comp = compress_data(raw)
            return comp
        except Exception:
            return None

    def forward(self, tokens: List[str], embeddings: np.ndarray) -> Dict:
        groups = self.group_tokens(tokens)
        # Cluster groups
        clusters = self.cluster_groups(groups)
        group_embs = []
        start = 0
        compressed_groups = []
        for g in clusters:
            size = len(g)
            emb = embeddings[start:start+size]
            group_embs.append(emb)
            # Compress group context
            comp = self.compress_group_context(emb)
            compressed_groups.append(comp)
            start += size
        # Intra-group
        intra = [self.attend_within_group(g) for g in group_embs]
        # Inter-group
        inter = self.attend_between_groups(intra)
        return {
            "groups": clusters,
            "intra": intra,
            "inter": inter,
            "edges": self.build_graph_edges(clusters),
            "compressed_groups": compressed_groups
        }

# Singleton
_hga = HierarchicalGraphAttention()

def process(tokens, embeddings):
    return _hga.forward(tokens, embeddings)
