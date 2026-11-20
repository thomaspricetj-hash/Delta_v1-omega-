"""
Weave Transformer – neural grid, branches, collapse, multi-scale context
"""
import torch
import torch.nn as nn

class WeaveBranch(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads=8, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(dim, 4*dim),
            nn.GELU(),
            nn.Linear(4*dim, dim),
        )
    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = x + attn_out
        x = x + self.ff(x)
        return x

class WeaveBranches(nn.Module):
    def __init__(self, dim: int, num_branches: int = 4):
        super().__init__()
        self.branches = nn.ModuleList([WeaveBranch(dim) for _ in range(num_branches)])
    def forward(self, x):
        return torch.stack([b(x) for b in self.branches], dim=1)

class WeaveCollapse(nn.Module):
    def __init__(self, dim: int, num_branches: int):
        super().__init__()
        self.proj = nn.Linear(num_branches * dim, dim)
    def forward(self, branch_outputs):
        b, nb, s, d = branch_outputs.shape
        x = branch_outputs.view(b, s, nb*d)
        return self.proj(x)

class WeaveGrid(nn.Module):
    def __init__(self, dim: int, grid_h: int, grid_w: int):
        super().__init__()
        self.grid_h, self.grid_w = grid_h, grid_w
        self.proj = nn.Linear(dim, dim)
    def forward(self, x):
        b, s, d = x.shape
        g = self.grid_h * self.grid_w
        grid = torch.zeros(b, g, d, device=x.device)
        grid[:, :s, :] = self.proj(x)
        return grid.view(b, self.grid_h, self.grid_w, d)

class ResonanceMemory(nn.Module):
    def __init__(self, dim: int, num_slots: int, kv_webb=None, persist_path=None):
        super().__init__()
        self.memory = nn.Parameter(torch.zeros(num_slots, dim))
        self.alpha = 0.9
        self.kv_webb = kv_webb
        self.persist_path = persist_path
        self.slot_keys = [f"resonance_slot_{i}" for i in range(num_slots)]
        if persist_path:
            self.load(persist_path)
    
    def save(self, path=None):
        p = path or self.persist_path
        if not p:
            return
        import torch
        torch.save(self.memory.detach().cpu(), p)
    
    def load(self, path):
        import torch, os
        if os.path.exists(path):
            state = torch.load(path, map_location='cpu')
            if state.shape == self.memory.shape:
                self.memory.data.copy_(state)
    @torch.no_grad()
    def update(self, x):
        mean_x = x.mean(dim=1).mean(dim=0)
        self.memory.data = self.alpha * self.memory.data + (1 - self.alpha) * mean_x
        # Persist to disk
        if self.persist_path:
            self.save()
        # Sync to KV-Webb
        if self.kv_webb:
            try:
                for i, key in enumerate(self.slot_keys[:1]):
                    self.kv_webb.store(key, {"mean": mean_x.detach().cpu().tolist(), "alpha": self.alpha})
            except Exception:
                pass
    def forward(self, x):
        # Update on forward pass
        self.update(x)
        return self.memory

def build_weave_pipeline(dim=256, num_branches=4, grid_h=16, grid_w=16, kv_webb=None, persist_path=None):
    branches = WeaveBranches(dim, num_branches)
    collapse = WeaveCollapse(dim, num_branches)
    grid = WeaveGrid(dim, grid_h, grid_w)
    resonance = ResonanceMemory(dim, num_slots=128, kv_webb=kv_webb, persist_path=persist_path)
    return {"branches": branches, "collapse": collapse, "grid": grid, "resonance": resonance}
