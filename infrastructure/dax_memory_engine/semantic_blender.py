"""
Semantic Blender – weighted delta mixing
"""
import torch
from typing import List

class SemanticBlender:
    def __init__(self, engine):
        self.engine = engine
    
    def blend(self, delta_names: List[str], weights: List[float]):
        # normalize weights
        total = sum(weights)
        weights = [w/total for w in weights]
        blended = {}
        for name, w in zip(delta_names, weights):
            delta = self.engine.delta_sets.get(name, {})
            for k, v in delta.items():
                if k not in blended:
                    blended[k] = torch.zeros_like(v)
                blended[k] += v * w
        return blended
