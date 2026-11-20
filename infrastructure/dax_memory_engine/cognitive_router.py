"""
Cognitive Memory Router – mode switching
"""
try:
    from .overlay_engine import DAXWeightOverlayEngine
except Exception:
    from overlay_engine import DAXWeightOverlayEngine

class CognitiveMemoryRouter:
    def __init__(self, engine: DAXWeightOverlayEngine):
        self.engine = engine
        self.modes = {}
    
    def register_mode(self, mode_name, *delta_names):
        self.modes[mode_name] = list(delta_names)
    
    def run_with_mode(self, mode_name, input_ids):
        deltas = self.modes.get(mode_name, [])
        with self.engine.apply_deltas(deltas):
            # placeholder inference
            return {"mode": mode_name, "deltas": deltas, "input": input_ids}
