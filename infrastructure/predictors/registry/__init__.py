"""
Auto-registry for predictors.
Imports all predictors and wraps them with Bloom filter cache, reward logging, and helpers.
"""
import importlib
import pkgutil
from pathlib import Path

from infrastructure.predictors.predictor_wrapper import wrap_predictor

PREDICTOR_MODULES = [
    "infrastructure.predictors.math_predictor",
    "infrastructure.predictors.code_exec_fast_predictor",
    "infrastructure.predictors.coding_predictor",
    "infrastructure.predictors.analogical_predictor",
    "infrastructure.predictors.temporal_predictor",
    "infrastructure.predictors.uncertainty_predictor",
    "infrastructure.predictors.rulebook_predictor",
    "infrastructure.predictors.cross_connect_grid_predictor",
    "infrastructure.predictors.master_overlay_predictor",
    "infrastructure.predictors.weave_predictor",
]

REGISTRY = {}

def _wrap_module(mod_path, name):
    try:
        mod = importlib.import_module(mod_path)
        # Find a predict function
        predict_fn = getattr(mod, "predict", None)
        if predict_fn is None:
            # Try common names
            for attr in ["predict_fn", "run", "forward"]:
                if hasattr(mod, attr):
                    predict_fn = getattr(mod, attr)
                    break
        if predict_fn and callable(predict_fn):
            wrapped = wrap_predictor(predict_fn, name)
            REGISTRY[name] = wrapped
            return wrapped
    except Exception as e:
        # Silently skip modules that can't be wrapped
        pass
    return None

for mod_path in PREDICTOR_MODULES:
    name = mod_path.split(".")[-1]
    _wrap_module(mod_path, name)

__all__ = ["REGISTRY"]
