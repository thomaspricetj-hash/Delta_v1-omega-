"""
Symbolic-Neural Fusion Layer
"""
import torch
try:
    from .pattern_graph import PatternGraph, BD3DCompressor, PTS, StabilityRules
    from .dax_encoder import DAXEncoder
except Exception:
    from pattern_graph import PatternGraph, BD3DCompressor, PTS, StabilityRules
    from dax_encoder import DAXEncoder

class SymbolicNeuralFusion:
    def __init__(self, dim=256, vocab_size=512, persist_path=None):
        self.pattern_graph = PatternGraph(persist_path)
        self.bd3d = BD3DCompressor()
        self.dax = DAXEncoder(vocab_size, dim)
        self.stability_rules = StabilityRules(min_freq=2, max_nodes=10000)
        self.persist_path = persist_path
    
    def fuse(self, weave_out: torch.Tensor):
        # weave_out: [batch, seq, dim]
        deltas = weave_out[:, 1:, :] - weave_out[:, :-1, :]
        pts_list = self.bd3d.compress(deltas)
        for pts in pts_list:
            self.pattern_graph.add_pattern(pts)
        # Build transitions
        for i in range(len(pts_list)-1):
            self.pattern_graph.add_transition(pts_list[i].pattern_id, pts_list[i+1].pattern_id)
        # Apply stability rules
        pruned = self.stability_rules.apply(self.pattern_graph)
        embeddings = self.dax.encode_pts(pts_list)
        if self.persist_path:
            self.pattern_graph.save()
        return {"pts": pts_list, "embeddings": embeddings, "graph": self.pattern_graph, "pruned": pruned}
