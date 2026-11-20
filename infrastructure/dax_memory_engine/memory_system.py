"""
MemorySystem – unified high-level API
"""
try:
    from .overlay_engine import DAXWeightOverlayEngine
    from .cognitive_router import CognitiveMemoryRouter
    from .semantic_blender import SemanticBlender
    from .drift_corrector import DriftCorrector
except Exception:
    from overlay_engine import DAXWeightOverlayEngine
    from cognitive_router import CognitiveMemoryRouter
    from semantic_blender import SemanticBlender
    from drift_corrector import DriftCorrector

class MemorySystem:
    def __init__(self, model, persist_dir=None):
        self.engine = DAXWeightOverlayEngine(model, persist_dir)
        self.router = CognitiveMemoryRouter(self.engine)
        self.blender = SemanticBlender(self.engine)
        self.corrector = DriftCorrector(self.engine)
    
    def register_delta(self, name, delta_dict):
        self.engine.register_delta(name, delta_dict)
    
    def register_mode(self, mode_name, *delta_names):
        self.router.register_mode(mode_name, *delta_names)
    
    def run_with_mode(self, mode_name, input_ids):
        return self.router.run_with_mode(mode_name, input_ids)
    
    def run_with_delta(self, delta_name, input_ids):
        with self.engine.apply_delta(delta_name):
            return {"delta": delta_name, "input": input_ids}
    
    def run_with_blend(self, delta_names, weights, input_ids):
        blended = self.blender.blend(delta_names, weights)
        # temporary apply
        with self.engine.apply_deltas(delta_names):
            return {"blended": delta_names, "weights": weights, "input": input_ids}
