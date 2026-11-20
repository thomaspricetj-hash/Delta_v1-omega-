"""
Production-grade multilayered window indexer with ledger for Translation OS v1
"""
import os
import json
import hashlib
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
try:
    from infrastructure.compression.mtd_codec import MTDPTSCodec
    MTD_AVAILABLE = True
except Exception:
    MTD_AVAILABLE = False
    MTDPTSCodec = None

try:
    from infrastructure.memory.qre_router import QREWindowPool
    QRE_AVAILABLE = True
except Exception:
    QRE_AVAILABLE = False
    QREWindowPool = None

try:
    import lmdb
    LMDB_AVAILABLE = True
except ImportError:
    LMDB_AVAILABLE = False

class WindowLedger:
    # Compatibility shim that delegates to MultilayeredWindowLedger
    def __init__(self, path: Path):
        from .multilayered_ledger import MultilayeredWindowLedger
        self.path = Path(path)
        self._ml = MultilayeredWindowLedger(self.path.parent)
        self.data = {
            'windows': {},
            'doc_index': {},
            'stats': {},
            'curriculum': {}
        }
    
    def save(self):
        self._ml.save_all()
    
    def record_sample(self, window_id: str, loss: float, step: int):
        self._ml.record_sample(window_id, loss, step)

class WindowIndexer:
    def __init__(self, base_path: Path, scales: List[int] = [128,256,512,1024]):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.scales = scales
        self.ledger_path = self.base_path / 'ledger.json'
        self.ledger = WindowLedger(self.ledger_path)
        self.env = None
        if LMDB_AVAILABLE:
            self.env = lmdb.open(str(self.base_path / 'windows.lmdb'), map_size=1e12)
        self.qre_pool = QREWindowPool() if QRE_AVAILABLE else None
        # DAX Registry + Hybrid BitDrop compression for delta storage
        try:
            from infrastructure.dax_registry.dax_registry import DAXRegistry
            self.dax_registry = DAXRegistry(self.base_path / 'dax')
            self.dax_enabled = True
        except Exception:
            self.dax_registry = None
            self.dax_enabled = False
    
    def build_from_texts(self, texts: List[str], tokenizer):
        """Pre-tokenize and build multilayered window index with delta compression"""
        doc_id = 0
        mtd_codec = MTDPTSCodec() if MTD_AVAILABLE else None
        for text in texts:
            text_clean = ''.join(ch if ord(ch) < 128 else ' ' for ch in text)
            ids = tokenizer.encode(text_clean)
            if len(ids) < 2:
                continue
            for scale in self.scales:
                stride = scale // 2
                n_windows = max(1, (len(ids) - scale) // stride + 1)
                for i in range(n_windows):
                    start = i * stride
                    window_ids = ids[start:start+scale]
                    if len(window_ids) < scale:
                        continue
                    window_id = f"{doc_id}_{scale}_{i}"
                    # MTD+PTS compression
                    if mtd_codec:
                        payload = mtd_codec.encode(window_ids)
                    else:
                        payload = np.array(window_ids, dtype=np.int32).tobytes()
                    # QRE allocation tracking
                    if self.qre_pool:
                        self.qre_pool.allocate_window(window_id, len(payload))
                    # DAX Registry + Hybrid BitDrop compression
                    if self.dax_enabled and self.dax_registry:
                        self.dax_registry.add_delta(window_id, payload, is_dax=True)
                    # Store window
                    if self.env:
                        with self.env.begin(write=True) as txn:
                            txn.put(window_id.encode(), payload)
                    # Update multilayered ledger
                    meta = {
                        'doc_id': doc_id,
                        'scale': scale,
                        'start': start,
                        'length': len(window_ids)
                    }
                    # Use multilayered ledger if available
                    if hasattr(self.ledger, '_ml'):
                        self.ledger._ml.record_window(window_id, meta, doc_id, scale)
                    else:
                        self.ledger.data['windows'][window_id] = meta
                        self.ledger.data['doc_index'].setdefault(doc_id, []).append(window_id)
                        self.ledger.data['curriculum'].setdefault(str(scale), []).append(window_id)
            doc_id += 1
        self.ledger.save()
    
    def sample_windows(self, scale: int, batch_size: int) -> List[np.ndarray]:
        """Sample windows for training"""
        # Use multilayered ledger if available
        if hasattr(self.ledger, '_ml'):
            window_ids = self.ledger._ml.get_curriculum_ids(scale)
        else:
            scale_key = str(scale)
            window_ids = self.ledger.data['curriculum'].get(scale_key, [])
        if not window_ids:
            return []
        # Cap window pool to avoid huge random.choice on massive ledger
        MAX_POOL = 1_000_000
        if len(window_ids) > MAX_POOL:
            # Randomly subsample pool once per call
            window_ids = np.random.choice(window_ids, size=MAX_POOL, replace=False).tolist()
        n = len(window_ids)
        size = min(batch_size, n)
        if size <= 0:
            return []
        chosen = np.random.choice(window_ids, size=size, replace=False) if n > 0 else []
        windows = []
        mtd_codec = MTDPTSCodec() if MTD_AVAILABLE else None
        if self.env:
            with self.env.begin() as txn:
                for wid in chosen:
                    data = txn.get(wid.encode())
                    if data:
                        if mtd_codec:
                            tokens = mtd_codec.decode(data)
                            windows.append(np.array(tokens, dtype=np.int32))
                        else:
                            arr = np.frombuffer(data, dtype=np.int32)
                            windows.append(arr)
        return windows
    
    def get_window(self, window_id: str) -> np.ndarray:
        if not self.env:
            return None
        mtd_codec = MTDPTSCodec() if MTD_AVAILABLE else None
        with self.env.begin() as txn:
            data = txn.get(window_id.encode())
            if not data:
                return None
            if mtd_codec:
                tokens = mtd_codec.decode(data)
                return np.array(tokens, dtype=np.int32)
            else:
                arr = np.frombuffer(data, dtype=np.int32)
                # legacy delta format
                flag = arr[0]
                payload = arr[1:]
                return payload
        return None

# Example usage
if __name__ == '__main__':
    from infrastructure.tokenizer.bpe_tokenizer import get_bpe_tokenizer
    base = Path('runtime/index')
    indexer = WindowIndexer(base)
    print('Indexer ready')
