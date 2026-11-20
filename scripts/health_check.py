"""
Quick health check for Translation OS v1
"""
import sys
from pathlib import Path
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

print("=== Translation OS v1 Health Check ===")
# 1. Music mapper
try:
    from infrastructure.music_mapper.music_mapper import get_harmony_summary
    h = get_harmony_summary()
    print(f"Music notes: {h.get('total_notes',0)} | harmony_score: {h.get('harmony_score')}")
except Exception as e:
    print(f"Music mapper error: {e}")

# 2. Predictor registry
try:
    from infrastructure.predictors.registry import REGISTRY
    print(f"Wrapped predictors: {len(REGISTRY)} -> {list(REGISTRY.keys())}")
except Exception as e:
    print(f"Predictor registry error: {e}")

# 3. Token predictor
try:
    from infrastructure.token_predictor.token_predictor import _predictor_cache
    stats = _predictor_cache.stats()
    print(f"Token predictor cache hit_rate: {stats['hit_rate']:.2f} hits={stats['hits']} misses={stats['misses']}")
except Exception as e:
    print(f"Token predictor error: {e}")

# 4. Weave decoder
try:
    from infrastructure.native_generator.weave_decoder import generate_with_weave
    out = generate_with_weave("Hello", max_new_tokens=5)
    print(f"Weave decoder alive, sample length {len(out)}")
except Exception as e:
    print(f"Weave decoder error: {e}")

print("=== Done ===")
