"""
DAX Weight Overlay Engine – core delta management
"""
from contextlib import contextmanager
from typing import Dict
import torch

class DAXWeightOverlayEngine:
    def __init__(self, model, persist_dir=None):
        self.model = model
        self.delta_sets: Dict[str, Dict[str, torch.Tensor]] = {}
        self.persist_dir = persist_dir
        if persist_dir:
            self.load_all(persist_dir)
    
    def register_delta(self, name: str, delta_dict: Dict[str, torch.Tensor]):
        self.delta_sets[name] = delta_dict
        if self.persist_dir:
            self.save_delta(name, delta_dict)
    
    def save_delta(self, name, delta_dict):
        import os
        os.makedirs(self.persist_dir, exist_ok=True)
        # Save metadata
        meta = {k: v.shape for k,v in delta_dict.items()}
        torch.save(delta_dict, os.path.join(self.persist_dir, f"{name}.pt"))
    
    def load_all(self, persist_dir):
        import os
        if not os.path.isdir(persist_dir):
            return
        for fname in os.listdir(persist_dir):
            if fname.endswith(".pt"):
                name = fname[:-3]
                path = os.path.join(persist_dir, fname)
                try:
                    delta = torch.load(path, map_location='cpu')
                    self.delta_sets[name] = delta
                except Exception:
                    continue
    
    @contextmanager
    def apply_delta(self, name: str):
        delta = self.delta_sets.get(name)
        if not delta:
            yield
            return
        # save original
        backups = {}
        for n, p in self.model.named_parameters():
            if n in delta:
                backups[n] = p.data.clone()
                p.data.add_(delta[n])
        try:
            yield
        finally:
            for n, p in self.model.named_parameters():
                if n in backups:
                    p.data.copy_(backups[n])
    
    @contextmanager
    def apply_deltas(self, names):
        backups = {}
        for name in names:
            delta = self.delta_sets.get(name)
            if not delta:
                continue
            for n, p in self.model.named_parameters():
                if n in delta:
                    if n not in backups:
                        backups[n] = p.data.clone()
                    p.data.add_(delta[n])
        try:
            yield
        finally:
            for n, p in self.model.named_parameters():
                if n in backups:
                    p.data.copy_(backups[n])
