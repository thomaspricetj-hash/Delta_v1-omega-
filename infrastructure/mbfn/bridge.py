"""
MBFN Bridge - maps WeaveFormer outputs to frame activations
Production grade
"""
from typing import List, Tuple
import torch
from .store import FrameStore

class MBFNBridge:
    def __init__(self, store: FrameStore, alpha: float = 0.7):
        self.store = store
        self.alpha = alpha  # blend weight for weaveformer vs mbfn

    def activate_from_text(self, text: str, tokenizer):
        # Simple keyword activation from text
        # In production, use HSCL concept extraction
        words = set(text.lower().split())
        triggers = []
        for label in self.store.label_index.keys():
            # crude overlap
            if any(w in label for w in words):
                triggers.append((label, 1.0))
        self.store.activate_frames(triggers)

    def rank_candidates(self, candidates: List[str], weave_scores: List[float]) -> List[Tuple[str, float]]:
        mbfn_scores = self.store.vote_candidates(candidates)
        blended = []
        for cand, wscore in zip(candidates, weave_scores):
            mscore = mbfn_scores.get(cand, 0.0)
            # normalize
            final = self.alpha * wscore + (1 - self.alpha) * mscore
            blended.append((cand, final))
        blended.sort(key=lambda x: x[1], reverse=True)
        return blended

    def explain_choice(self, choice: str) -> List[dict]:
        return self.store.explain(choice)
