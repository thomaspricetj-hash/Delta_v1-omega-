"""
BitDrop v2 wrapper for all persistence
"""
import sys
from pathlib import Path
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "infrastructure"))
from bitdrop_v2_integration import compress_data, bitdrop_v2_status

def compress_and_save(path, data_bytes):
    # For now just return compressed bytes, actual save handled by caller
    return compress_data(data_bytes)

def is_ready():
    return bitdrop_v2_status()
