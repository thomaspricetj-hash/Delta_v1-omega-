"""
Wire All Integration Module
Production-grade wiring of Structural Learning Engine, MDWE, DAX Registry + Hybrid BitDrop
for TRANSLATION_OS_V1
"""
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import json
import time

from infrastructure.structural.learner_bridge import LearnerBridge, EpisodeTrace
from infrastructure.mdwe_engine.mdwe_core import MDWEEngine
from infrastructure.dax_registry.dax_registry import DAXRegistry

class IntegratedCognitiveCore:
    def __init__(self, base_dir: Path = Path("runtime")):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Structural Learning Engine
        self.structural = LearnerBridge(journal_dir=self.base_dir / "structural" / "journal")
        
        # MDWE Engine
        self.mdwe = MDWEEngine(num_neurons=1024, dim=256)
        
        # DAX Registry with Hybrid BitDrop
        self.dax = DAXRegistry(self.base_dir / "dax_registry")
        
        # BitDrop log writer for all logs
        from infrastructure.logging.bitdrop_log_writer import BitDropLogWriter
        self.log_writer = BitDropLogWriter(self.base_dir / "logs")
        
        # State
        self.step = 0
        self.master_state = np.zeros(1024, dtype=np.float32).tobytes()
        
    def process_episode(self, tokens: List[str], activations: np.ndarray, tags: List[str] = None):
        tags = tags or []
        ep = EpisodeTrace(tokens=tokens, activations=[activations.tolist()], tags=tags, time=time.time())
        
        # 1. Structural Learning
        self.structural.learner.learn_from(ep)
        
        # 2. MDWE forward with heat gating
        # Use activations as input to MDWE
        input_vec = activations[:self.mdwe.dim] if activations.size >= self.mdwe.dim else np.pad(activations, (0, self.mdwe.dim - activations.size))
        out, active_ids = self.mdwe.forward(input_vec)
        
        # Record delta graph ledger
        scores = self.mdwe.router.heat_extract(input_vec)
        self.mdwe.ledger.record(active_ids, scores)
        
        # 3. DAX delta journaling with BitDrop compression
        delta_payload = {
            "step": self.step,
            "tokens": tokens,
            "active_neurons": active_ids,
            "mdwe_out_norm": float(np.linalg.norm(out)),
            "structural_version": self.structural.learner.version
        }
        delta_bytes = json.dumps(delta_payload).encode('utf-8')
        delta_id = f"ep_{self.step}_{int(time.time()*1000)}"
        self.dax.add_delta(delta_id, delta_bytes, is_dax=True)
        
        # 4. Update master state via XOR delta
        # Simulate master weight update
        new_state = np.frombuffer(self.master_state, dtype=np.float32).copy()
        # Simple delta: add MDWE output to first dim
        delta_state = np.zeros_like(new_state)
        delta_state[:len(out)] = out[:len(new_state)]
        delta_state_bytes = delta_state.tobytes()
        # XOR with current master
        self.master_state = self.dax.reconstruct_state(delta_id, self.master_state) if False else self._xor_states(self.master_state, delta_state_bytes)
        
        self.step += 1
        return {
            "structural_version": self.structural.learner.version,
            "mdwe_active": len(active_ids),
            "mdwe_out_norm": float(np.linalg.norm(out)),
            "delta_id": delta_id
        }
    
    def _xor_states(self, a: bytes, b: bytes):
        max_len = max(len(a), len(b))
        a_padded = a + b'\x00'*(max_len-len(a))
        b_padded = b + b'\x00'*(max_len-len(b))
        return bytes(x ^ y for x, y in zip(a_padded, b_padded))
    
    def save_checkpoint(self, path: Path):
        # Save structural learner
        self.structural.learner.save_state(path.with_suffix('.structural.json'))
        # Save MDWE ledger
        ledger = self.mdwe.ledger.export()
        with open(path.with_suffix('.mdwe.json'), 'w') as f:
            json.dump(ledger, f)
        # Save DAX stats
        stats = self.dax.stats()
        with open(path.with_suffix('.dax_stats.json'), 'w') as f:
            json.dump(stats, f)
    
    def health_check(self):
        return {
            "structural_version": self.structural.learner.version,
            "mdwe_neurons": self.mdwe.num_neurons,
            "dax_deltas": len(self.dax.deltas),
            "step": self.step
        }

def create_integrated_core(base_dir: Path = Path("runtime")) -> IntegratedCognitiveCore:
    return IntegratedCognitiveCore(base_dir)
