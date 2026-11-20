"""
Lazy loading WeaveFormer with BitDrop v2 compression and auto-offload
Keeps VRAM minimal without precision loss
"""
import torch
import torch.nn as nn
from infrastructure.true_hybrid.dual_weight_layers import DualWeightWeaveFormer

class LazyWeaveFormer(nn.Module):
    def __init__(self, ckpt_path, device='cuda', offload_after_steps=5):
        super().__init__()
        self.device = device
        self.offload_after_steps = offload_after_steps
        self.step_counter = {}
        
        # Load checkpoint metadata on CPU only
        ckpt = torch.load(ckpt_path, map_location='cpu')
        state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
        
        dim = int(state['embed.weight'].shape[1])
        vocab = int(state['embed.weight'].shape[0])
        self.model = DualWeightWeaveFormer(dim=dim, num_layers=32, vocab_size=vocab,
                                          use_dual_weights=True, nhead=9)
        self.model.to('cpu')
        self.state_dict = state
        self.loaded_layers = set()
        
    def _load_layer(self, name):
        if name in self.loaded_layers:
            return
        # Load only this layer's parameters
        layer_state = {k.replace(name+'.',''): v for k,v in self.state_dict.items() 
                       if k.startswith(name+'.')}
        layer = dict(self.model.named_modules())[name]
        layer.load_state_dict(layer_state, strict=False)
        layer.to(self.device)
        self.loaded_layers.add(name)
        self.step_counter[name] = 0
    
    def _maybe_offload(self, name):
        if name not in self.loaded_layers:
            return
        self.step_counter[name] += 1
        if self.step_counter[name] > self.offload_after_steps:
            layer = dict(self.model.named_modules())[name]
            layer.to('cpu')
            self.loaded_layers.discard(name)
    
    def forward(self, ids):
        x = self.model.embed(ids).to(self.device)
        for i, layer in enumerate(self.model.layers):
            name = f'layers.{i}'
            self._load_layer(name)
            x = layer(x)
            self._maybe_offload(name)
        x = self.model.norm(x)
        self._load_layer('head')
        logits = self.model.head(x)
        return logits
