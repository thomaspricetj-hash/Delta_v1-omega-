"""
DAX Adapter for real model integration
Wraps a PyTorch model with MemorySystem overlays
"""
import torch.nn as nn
from infrastructure.dax_memory_engine.memory_system import MemorySystem

class DAXAdapter(nn.Module):
    def __init__(self, base_model, persist_dir=None):
        super().__init__()
        self.base = base_model
        self.memory = MemorySystem(base_model, persist_dir=persist_dir)
    
    def register_delta(self, name, delta_dict):
        self.memory.register_delta(name, delta_dict)
    
    def forward(self, *args, mode=None, delta=None, blend=None, **kwargs):
        if mode:
            # Use cognitive router mode
            # For demo, just run base model under mode context
            # In real use, overlay engine would modify weights during forward
            return self.base(*args, **kwargs)
        return self.base(*args, **kwargs)
