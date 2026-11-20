"""
Demo integration of DeltaVectorMemory with WeaveFormer training loop.
Shows how to use delta-backed vector memory as AI memory storage.
"""
from pathlib import Path
import numpy as np
from infrastructure.memory.vector_store import DeltaVectorMemory

def demo():
    mem = DeltaVectorMemory(dim=256, ledger_path=Path("runtime/memory/ledger.jsonl"))
    # Simulate adding window embeddings
    for i in range(10):
        vec = np.random.randn(256).astype(np.float32)
        mem.add_memory(f"win_{i}", vec)
        if i > 0:
            mem.link_memories(f"win_{i-1}", f"win_{i}", weight=0.9, layer=2)
    # Update with delta
    delta = np.random.randn(256).astype(np.float32) * 0.01
    mem.update_memory("win_5", delta)
    # Query
    query = np.random.randn(256).astype(np.float32)
    results = mem.query(query, k=3)
    print("Top results:", results)
    # Persist
    mem.save(Path("runtime/memory/vecmem"))
    # Reload
    mem2 = DeltaVectorMemory(dim=256)
    mem2.load(Path("runtime/memory/vecmem"))
    print("Reloaded vectors:", len(mem2.store.vectors))

if __name__ == "__main__":
    demo()
