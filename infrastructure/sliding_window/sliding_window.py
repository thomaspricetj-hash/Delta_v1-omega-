"""
Global sliding window for orchestrator
"""
from collections import deque
from pathlib import Path

class SlidingWindow:
    def __init__(self, size=4096):
        self.size = size
        # Multi-dimensional windows: query, response, concept, metadata, dialogue
        self.windows = {
            'query': deque(maxlen=size),
            'response': deque(maxlen=size),
            'concept': deque(maxlen=size),
            'meta': deque(maxlen=size),
            'dialogue': deque(maxlen=size)
        }
    
    def add(self, item, dim='query'):
        if dim in self.windows:
            self.windows[dim].append(item)
    
    def get(self, dim=None):
        if dim:
            return list(self.windows.get(dim, []))
        # Return merged view
        merged = []
        for d in ['query','response','concept','meta','dialogue']:
            merged.extend(list(self.windows[d]))
        return merged
    
    def clear(self):
        for w in self.windows.values():
            w.clear()

# Global instance
_window = SlidingWindow(size=4096)

def get_window():
    return _window

def add_to_window(item, dim='query'):
    _window.add(item, dim)

def get_window_contents():
    return _window.get()
