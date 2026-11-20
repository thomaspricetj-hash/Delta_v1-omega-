"""
MBFN Frame Store - production grade sparse activation, voting, persistence
"""
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple
from .frame import Frame

class FrameStore:
    def __init__(self):
        self.frames: Dict[str, Frame] = {}
        self.label_index: Dict[str, str] = {}
        self.layer_index: Dict[str, Set[str]] = {}

    def add_frame(self, frame: Frame):
        self.frames[frame.id] = frame
        self.label_index[frame.label.lower()] = frame.id
        self.layer_index.setdefault(frame.layer, set()).add(frame.id)

    def get_by_label(self, label: str) -> Frame:
        fid = self.label_index.get(label.lower())
        return self.frames.get(fid) if fid else None

    def activate_frames(self, triggers: List[Tuple[str, float]]):
        # triggers = [(label, evidence), ...]
        for label, evidence in triggers:
            frame = self.get_by_label(label)
            if frame:
                frame.activate(evidence)
                # propagate to connections with decay
                for conn_id in frame.connections:
                    conn = self.frames.get(conn_id)
                    if conn:
                        conn.activate(evidence * 0.5)

    def vote_candidates(self, candidate_labels: List[str]) -> Dict[str, float]:
        scores = {c: 0.0 for c in candidate_labels}
        # only active frames vote
        active = [f for f in self.frames.values() if f.state != 0]
        for frame in active:
            # simple heuristic: frame votes for candidates sharing label tokens
            for cand in candidate_labels:
                if frame.label.lower() in cand.lower() or cand.lower() in frame.label.lower():
                    scores[cand] += frame.vote(cand)
        return scores

    def explain(self, candidate: str) -> List[Dict]:
        explanation = []
        for frame in self.frames.values():
            if frame.state == 0:
                continue
            if frame.label.lower() in candidate.lower():
                explanation.append({
                    "frame": frame.label,
                    "layer": frame.layer,
                    "state": "SUPPORT" if frame.state > 0 else "OPPOSE",
                    "strength": frame.strength
                })
        explanation.sort(key=lambda x: x["strength"], reverse=True)
        return explanation

    def save(self, path: Path):
        path = Path(path)
        data = {
            "frames": [f.to_dict() for f in self.frames.values()]
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    def load(self, path: Path):
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.frames.clear()
        self.label_index.clear()
        self.layer_index.clear()
        for fdict in data.get("frames", []):
            frame = Frame(
                id=fdict["id"],
                label=fdict["label"],
                layer=fdict["layer"],
                state=fdict["state"],
                strength=fdict["strength"],
                connections=set(fdict["connections"]),
                metadata=fdict.get("metadata", {})
            )
            self.add_frame(frame)

    def build_from_hscl(self, hscl_dict_path: Path):
        # Import HSCL numeric dictionary and create concept frames
        import json
        with open(hscl_dict_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        entries = data.get("entries", {})
        for phrase, meta in entries.items():
            frame = Frame.create(label=phrase, layer="Concept", metadata={"id": meta.get("id")})
            self.add_frame(frame)
        # Create generic feature frames
        for feat in ["Country", "Capital", "European", "Nation", "City"]:
            if not self.get_by_label(feat):
                self.add_frame(Frame.create(label=feat, layer="Feature"))
