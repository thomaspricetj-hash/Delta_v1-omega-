"""
Mixed precision helpers for CPU/Numpy.
Provides autocast-like context and loss scaling stub.
"""
import numpy as np

class autocast:
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

def cast_to_float16(arr):
    return arr.astype(np.float16)

def cast_to_float32(arr):
    return arr.astype(np.float32)
