"""
BitDrop v2 integration wrapper
Integrates the Rust BitDrop v2 engine into Translation OS design
"""
import os
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent.parent.parent / "TRANSLATION_OS_V1" / "bitdrop_v2"

try:
    import bitdrop_v2 as _bd2
    _BD2_AVAILABLE = True
except Exception:
    _BD2_AVAILABLE = False

class BitDropV2Engine:
    def __init__(self):
        self.model_path = BASE / "bitdrop_global_model.bin"
        self.showcase_path = BASE / "bitdrop_showcase.bin"
        self.ready = self.model_path.exists()
        self.rust_available = _BD2_AVAILABLE
        try:
            from pattern_classifier.pattern_classifier import classify, get_class_stats
            self.classifier_available = True
        except Exception:
            self.classifier_available = False
    
    def compress(self, data: bytes) -> dict:
        pattern_info = None
        if self.classifier_available:
            try:
                from pattern_classifier.pattern_classifier import classify, get_class_stats
                classified = classify(data)
                pattern_info = {
                    "patterns_classified": len(classified),
                    "class_stats": get_class_stats(),
                    "sample_patterns": classified[:10]
                }
            except Exception:
                pattern_info = {"error": "classifier_failed"}
        
        if self.rust_available:
            try:
                compressed = _bd2.compress_adaptive_py(data)
                fast = _bd2.compress_fast_py(data)
                return {
                    "engine": "BitDrop v2 Rust",
                    "model_loaded": self.ready,
                    "rust_available": True,
                    "compressed_len": len(compressed),
                    "fast_len": len(fast),
                    "original_len": len(data),
                    "ratio": len(compressed)/max(1,len(data)),
                    "source": "bitdrop_v2_rust",
                    "pattern_classifier": pattern_info
                }
            except Exception as e:
                return {"engine":"BitDrop v2","rust_error":str(e),"model_loaded":self.ready,"pattern_classifier":pattern_info}
        
        # Fallback placeholder
        try:
            sys_path = str(Path(__file__).resolve().parent.parent / "bitdrop_compression")
            import sys
            sys.path.insert(0, sys_path)
            from bitdrop_compression_v0 import compress_text
            text = data.decode('utf-8', errors='ignore')
            metrics = compress_text(text)
            return {
                "engine": "BitDrop v2",
                "model_loaded": self.ready,
                "rust_available": False,
                "metrics": metrics,
                "source": "bitdrop_v2_integration_fallback",
                "pattern_classifier": pattern_info
            }
        except Exception as e:
            return {"engine":"BitDrop v2","error":str(e),"model_loaded":self.ready,"pattern_classifier":pattern_info}

    def get_status(self):
        status = {
            "model_path": str(self.model_path),
            "model_exists": self.ready,
            "showcase_exists": self.showcase_path.exists(),
            "rust_available": self.rust_available
        }
        if self.rust_available:
            try:
                status["gpu_available"] = _bd2.gpu_available_py()
            except Exception:
                status["gpu_available"] = False
        return status

_engine = BitDropV2Engine()

def compress_data(data: bytes):
    return _engine.compress(data)

def bitdrop_v2_status():
    return _engine.get_status()
