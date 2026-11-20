"""
Training pipeline for WeaveFormer on pattern sequences
"""
import torch
from .weaveformer import WeaveFormer

def train_step(model, optimizer, ids, targets, step):
    logits = model(ids)
    loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
    optimizer.zero_grad()
    loss.backward()
    # Log weight updates as musical notes before step
    try:
        import sys
        from pathlib import Path
        BASE = Path(__file__).resolve().parents[2]
        sys.path.insert(0, str(BASE))
        from infrastructure.music_mapper.weight_mapper import log_weight_update, get_weight_harmony
        for name, param in model.named_parameters():
            if param.grad is not None:
                delta_norm = param.grad.norm().item()
                weight_norm = param.norm().item()
                log_weight_update(name, delta_norm, weight_norm, step)
        # Auto-stabilizer: modulate LR based on weight harmony
        harmony = get_weight_harmony()
        if harmony:
            score = harmony.get('harmony_score', 0.5)
            # Dissonance -> cool down LR, consonance -> warm up
            if score < 0.3:
                for g in optimizer.param_groups:
                    g['lr'] *= 0.85
            elif score > 0.7:
                for g in optimizer.param_groups:
                    g['lr'] = min(g['lr'] * 1.05, 1e-2)
    except Exception:
        pass
    optimizer.step()
    return loss.item()

def dummy_data_loader(batch_size=4, seq_len=32, vocab_size=512, steps=10):
    for _ in range(steps):
        ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))
        yield ids, targets

def train_weaveformer(dim=256, num_layers=2, vocab_size=512, steps=10):
    model = WeaveFormer(dim, num_layers, vocab_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    for step, (ids, targets) in enumerate(dummy_data_loader(steps=steps), start=1):
        loss = train_step(model, optimizer, ids, targets, step)
        losses.append(loss)
    return {"final_loss": losses[-1], "steps": len(losses)}

if __name__ == "__main__":
    print(train_weaveformer())
