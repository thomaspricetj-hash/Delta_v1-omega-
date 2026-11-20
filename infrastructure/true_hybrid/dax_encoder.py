"""
DAX Encoder – Delta-State encoding
"""
import torch
import torch.nn as nn
from dataclasses import dataclass

@dataclass
class DeltaState:
    deltas: torch.Tensor
    meta: dict

class DAXEncoder:
    def __init__(self, vocab_size: int, dim: int):
        self.embedding = nn.Embedding(vocab_size, dim)
    
    def encode_pts(self, pts_list):
        ids = torch.tensor([p.pattern_id for p in pts_list], dtype=torch.long)
        return self.embedding(ids)
    
    @staticmethod
    def build_delta_states(x: torch.Tensor):
        # x: [batch, seq, dim]
        deltas = x[:, 1:, :] - x[:, :-1, :]
        return [DeltaState(deltas[i], {"pos": i}) for i in range(deltas.shape[0])]
