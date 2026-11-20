"""
Generic wrapper to add Bloom filter cache, reward logging, helpers to any predictor.
"""
from infrastructure.predictors.predictor_base import PredictorCache, log_reward, log_predictor_event
import hashlib

def wrap_predictor(predict_fn, name):
    cache = PredictorCache(name)
    def wrapped(query, diagnostics=None):
        # Cache key
        key = hashlib.sha256((name + "|" + str(query)).encode()).hexdigest()
        cached_val, hit = cache.get(key)
        if hit:
            log_predictor_event(name, "cache_hit", {"key": key})
            return cached_val
        # Call original
        try:
            result = predict_fn(query)
        except Exception as e:
            log_predictor_event(name, "error", {"error": str(e)})
            return None
        if result:
            cache.set(key, result)
            conf = float(result.get("confidence", 0.5))
            # Simple reward shaping: reward = confidence * hit_bonus
            reward = conf * (1.2 if hit else 1.0)
            log_reward(name, reward, conf, query)
            log_predictor_event(name, "predict", {"confidence": conf})
        return result
    wrapped.cache = cache
    wrapped.name = name
    return wrapped
