"""
Chunk retriever for effective huge context window.
Splits text into 64-token chunks, retrieves relevant chunks via embeddings/weave.
Supports multi-roundabout retrieval and multi-layer expanding context.
Integrates BitDrop v2 compression to increase effective context capacity.
Auto-expanding layers with RAM monitor, delta encoding, and sliding window.
"""
import sys
from pathlib import Path
BASE = Path(__file__).resolve().parents[2]

try:
    import psutil
    _PSUTIL_AVAILABLE = True
except Exception:
    _PSUTIL_AVAILABLE = False

def chunk_text(text, chunk_size=64):
    tokens = text.split()
    chunks = []
    for i in range(0, len(tokens), chunk_size):
        chunks.append(" ".join(tokens[i:i+chunk_size]))
    return chunks

def retrieve_relevant(query, corpus_chunks, top_k=4):
    sys.path.insert(0, str(BASE / "infrastructure" / "embeddings"))
    from embeddings import embed
    import numpy as np
    q_vec = embed(query)
    scores = []
    for chunk in corpus_chunks:
        v = embed(chunk)
        score = float(np.dot(q_vec, v))
        scores.append((chunk, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    return [c for c,_ in scores[:top_k]]

def multi_round_retrieve(query, corpus_chunks, rounds=3, top_k=4):
    visited = set()
    current_query = query
    collected = []
    for r in range(rounds):
        rel = retrieve_relevant(current_query, corpus_chunks, top_k=top_k)
        new_items = []
        for c in rel:
            if c not in visited:
                visited.add(c)
                new_items.append(c)
        collected.extend(new_items)
        if new_items:
            current_query = query + " ".join(new_items[:top_k])
        if not new_items:
            break
    return collected

def _compress_context(text):
    try:
        sys.path.insert(0, str(BASE / "infrastructure" / "bitdrop_compression"))
        from bitdrop_compression_v0 import compress_text
        metrics = compress_text(text)
        return text, metrics
    except Exception:
        return text, {"compression_ratio": 1.0}

def _decompress_context(compressed_text):
    return compressed_text

class RAMMonitor:
    def __init__(self, max_mem_percent=75.0):
        self.max_mem_percent = max_mem_percent
    def current_mem_percent(self):
        if not _PSUTIL_AVAILABLE:
            return 0.0
        return psutil.virtual_memory().percent
    def can_expand(self):
        return self.current_mem_percent() < self.max_mem_percent
    def get_status(self):
        if not _PSUTIL_AVAILABLE:
            return {"psutil_available": False}
        vm = psutil.virtual_memory()
        return {
            "psutil_available": True,
            "percent": vm.percent,
            "available_mb": vm.available / (1024*1024),
            "used_mb": vm.used / (1024*1024)
        }

_ram_monitor = RAMMonitor(max_mem_percent=75.0)

def _delta_encode(prev_text, curr_text):
    prev_set = set(prev_text.split())
    curr_tokens = curr_text.split()
    delta_tokens = [t for t in curr_tokens if t not in prev_set]
    delta_text = " ".join(delta_tokens)
    delta_ratio = len(curr_text.split()) / max(1, len(delta_tokens))
    return delta_text, delta_ratio

def _sliding_windows(text, window_size=1024, stride=512):
    tokens = text.split()
    windows = []
    for i in range(0, len(tokens), stride):
        window = tokens[i:i+window_size]
        if not window:
            break
        windows.append(" ".join(window))
        if i + window_size >= len(tokens):
            break
    return windows

class LayeredContextBuilder:
    def __init__(self, layers=4, base_top_k=4, expand_factor=1.5, compress=True, auto_expand=True, use_delta=True, sliding_window=True, window_size=1024, stride=None):
        self.base_layers = layers
        self.base_top_k = base_top_k
        self.expand_factor = expand_factor
        self.compress = compress
        self.auto_expand = auto_expand
        self.use_delta = use_delta
        self.sliding_window = sliding_window
        self.base_window_size = window_size
        self.base_stride = stride

    def _dynamic_window_and_stride(self):
        ram_pct = _ram_monitor.current_mem_percent()
        # Window size grows with available RAM
        if ram_pct < 40:
            window_size = 2048
        elif ram_pct < 55:
            window_size = 1536
        elif ram_pct < 70:
            window_size = 1024
        else:
            window_size = 512
        # Stride scales with window
        if ram_pct < 50:
            stride = int(window_size * 0.75)
        elif ram_pct < 65:
            stride = window_size // 2
        else:
            stride = int(window_size * 0.9)
        return window_size, stride

    def _choose_layers(self):
        if not self.auto_expand:
            return self.base_layers
        layers = self.base_layers
        for target in range(self.base_layers+1, 9):
            if _ram_monitor.can_expand():
                layers = target
            else:
                break
        return layers

    def build(self, query, corpus_text):
        window_size, stride = self._dynamic_window_and_stride()
        chunks = chunk_text(corpus_text)
        layers = self._choose_layers()
        layer_texts = []
        layer_metrics = []
        visited = set()
        current_query = query
        prev_layer_text = ""
        for i in range(layers):
            if i > 0 and not _ram_monitor.can_expand():
                break
            top_k = int(self.base_top_k * (self.expand_factor ** i))
            rel = retrieve_relevant(current_query, chunks, top_k=top_k)
            new_items = []
            for c in rel:
                if c not in visited:
                    visited.add(c)
                    new_items.append(c)
            if not new_items:
                break
            layer_str = " ".join(new_items)
            if self.use_delta and i > 0:
                delta_text, delta_ratio = _delta_encode(prev_layer_text, layer_str)
                layer_str = delta_text
            else:
                delta_ratio = 1.0
            if self.compress:
                compressed, metrics = _compress_context(layer_str)
                metrics = dict(metrics)
                metrics["delta_ratio"] = delta_ratio
                layer_texts.append(compressed)
                layer_metrics.append(metrics)
            else:
                layer_texts.append(layer_str)
                layer_metrics.append({"compression_ratio":1.0, "delta_ratio":delta_ratio})
            prev_layer_text = " ".join(new_items)
            current_query = query + " ".join(new_items[:top_k])
        stacked = " [LAYER {}] ".format(layers-1).join(reversed(layer_texts))
        if self.sliding_window:
            stacked_windows = _sliding_windows(stacked, window_size=window_size, stride=stride)
            stacked_for_model = stacked_windows[0] if stacked_windows else stacked
            num_stacked_windows = len(stacked_windows)
        else:
            stacked_for_model = stacked
            num_stacked_windows = 1
        total_tokens = sum(len(t.split()) for t in layer_texts)
        avg_comp = sum(m.get("compression_ratio",1.0) for m in layer_metrics)/max(1,len(layer_metrics))
        avg_delta = sum(m.get("delta_ratio",1.0) for m in layer_metrics)/max(1,len(layer_metrics))
        effective_raw_tokens = int(total_tokens * avg_delta)
        ram_status = _ram_monitor.get_status()
        agg_metrics = {
            "layers": len(layer_texts),
            "requested_layers": layers,
            "total_stored_tokens": total_tokens,
            "effective_raw_tokens": effective_raw_tokens,
            "avg_compression_ratio": avg_comp,
            "avg_delta_ratio": avg_delta,
            "stacked_sliding_windows": num_stacked_windows,
            "window_size": window_size,
            "stride": stride,
            "ram_percent": ram_status.get("percent", None),
            "ram_available_mb": ram_status.get("available_mb", None),
            "layer_metrics": layer_metrics
        }
        return f"{query} [LAYERED_CONTEXT] {stacked_for_model}", agg_metrics

def build_effective_context(query, corpus_text, rounds=3):
    chunks = chunk_text(corpus_text)
    relevant = multi_round_retrieve(query, chunks, rounds=rounds, top_k=4)
    context_str = " ".join(relevant)
    compressed_context, metrics = _compress_context(context_str)
    decompressed = _decompress_context(compressed_context)
    return f"{query} [CONTEXT] {decompressed}", metrics

def build_layered_context(query, corpus_text, layers=4):
    builder = LayeredContextBuilder(layers=layers, auto_expand=True, use_delta=True, sliding_window=True, window_size=1024, stride=None)
    return builder.build(query, corpus_text)

def get_ram_status():
    return _ram_monitor.get_status()
