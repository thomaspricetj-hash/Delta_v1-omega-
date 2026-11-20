"""
Fractal Multi-Agent System v2.0 – Self-improving hierarchical multi-agent architecture
Simplified integration for Translation OS V1
"""
from dataclasses import dataclass, field
from typing import Dict, List, Any
import uuid
import time

@dataclass
class Cell:
    id: str
    task: str
    status: str = "pending"
    output: Any = None
    confidence: float = 0.0
    attempts: int = 0

class SharedIndex:
    def __init__(self):
        self.store = {}
    def add(self, key, value):
        self.store[key] = value
    def search(self, query):
        return [v for k,v in self.store.items() if query.lower() in str(k).lower()]

class ReviewRepairLoop:
    def __init__(self):
        self.semantic_feedback_used = False
        try:
            import sys
            from pathlib import Path
            BASE = Path(__file__).resolve().parent.parent.parent.parent
            sys.path.insert(0, str(BASE / "runtime"))
            from semantic_map import feedback as semantic_feedback, predict_semantic
            self.semantic_feedback = semantic_feedback
            self.predict_semantic = predict_semantic
        except Exception:
            self.semantic_feedback = None
            self.predict_semantic = None

    def review_cell(self, cell: Cell) -> bool:
        valid = cell.output is not None and str(cell.output).strip() != ""
        if not valid:
            # trigger semantic penalty for empty output
            if self.semantic_feedback:
                try:
                    self.semantic_feedback(cell.task, "<empty>", correct=False)
                    self.semantic_feedback_used = True
                except Exception:
                    pass
        return valid

    def repair(self, cell: Cell):
        cell.attempts += 1
        cell.status = "pending"
        # Try semantic prediction for repair hint
        if self.predict_semantic:
            try:
                pred = self.predict_semantic(cell.task, top_k=1)
                if pred and pred.get("candidates"):
                    hint = pred["candidates"][0]["answer"]
                    cell.output = f"[repair hint: {hint}]"
            except Exception:
                pass
        return cell

class HelperAgentManager:
    def __init__(self):
        self.helpers = {}
        self.micro_helpers = {}
        try:
            import sys
            from pathlib import Path
            BASE = Path(__file__).resolve().parent.parent.parent.parent
            sys.path.insert(0, str(BASE / "infrastructure" / "helpers"))
            from micro_helpers import is_definition, is_math, is_project
            self.is_definition = is_definition
            self.is_math = is_math
            self.is_project = is_project
        except Exception:
            self.is_definition = lambda q: False
            self.is_math = lambda q: False
            self.is_project = lambda q: False

    def create_helper(self, specialization: str):
        hid = str(uuid.uuid4())
        self.helpers[hid] = {"specialization": specialization, "load": 0}
        return hid
    
    def execute_task(self, task: str, specialization: str = "general"):
        # Route through micro-helpers for fast classification
        if self.is_definition(task):
            specialization = "definition"
        elif self.is_math(task):
            specialization = "math"
        elif self.is_project(task):
            specialization = "project"
        return f"Helper[{specialization}] executed: {task}"

class DataRoundabout:
    def __init__(self):
        self.direct = {}
        self.cache = {}
    def route_data(self, key, data=None):
        if data is not None:
            self.direct[key] = data
            return True
        return self.direct.get(key)

class ControlRoundabout:
    def __init__(self):
        self.decisions = []
    def route_decision(self, decision_type, context):
        self.decisions.append({"type": decision_type, "context": context})
        return True

class DualRoundaboutSystem:
    def __init__(self):
        self.data = DataRoundabout()
        self.control = ControlRoundabout()
        self.kv_webb = None
        self.memory_organizer = None
        try:
            import sys
            from pathlib import Path
            BASE = Path(__file__).resolve().parent.parent.parent.parent
            sys.path.insert(0, str(BASE / "infrastructure" / "kv_webb" / "kv_web_runtime"))
            from runtime import KvWebRuntime
            self.kv_webb = KvWebRuntime()
        except Exception:
            self.kv_webb = None
        try:
            sys.path.insert(0, str(BASE / "infrastructure" / "memory_organizer"))
            from memory_organizer import organize_memory
            self.memory_organizer = organize_memory
        except Exception:
            self.memory_organizer = None

    def route_with_memory(self, key, data):
        # Offload to KV-Webb and Memory Organizer
        if self.kv_webb:
            try:
                # simple write hook
                self.kv_webb.store(key, data)
            except Exception:
                pass
        if self.memory_organizer:
            try:
                self.memory_organizer(key)
            except Exception:
                pass
        return self.data.route_data(key, data)

class FractalSystemOrchestrator:
    def __init__(self):
        self.index = SharedIndex()
        self.review = ReviewRepairLoop()
        self.helpers = HelperAgentManager()
        self.roundabouts = DualRoundaboutSystem()
        self.cells: List[Cell] = []

    def submit_goal(self, goal: str):
        # Break into grid
        tasks = [f"{goal} part {i}" for i in range(1,5)]
        self.cells = [Cell(id=str(uuid.uuid4()), task=t) for t in tasks]

    def execute_goal(self):
        results = []
        for cell in self.cells:
            helper_id = self.helpers.create_helper("analytical")
            cell.output = self.helpers.execute_task(cell.task, "analytical")
            cell.status = "done"
            cell.confidence = 0.8
            # Review & Repair with semantic feedback
            if not self.review.review_cell(cell):
                cell = self.review.repair(cell)
                cell.output = self.helpers.execute_task(cell.task, "verification")
                # Penalize failed attempt in semantic map
                try:
                    from pathlib import Path
                    import sys
                    BASE = Path(__file__).resolve().parent.parent.parent.parent
                    sys.path.insert(0, str(BASE / "runtime"))
                    from semantic_map import feedback
                    feedback(cell.task, cell.output, correct=False)
                except Exception:
                    pass
            else:
                # Reward successful execution
                try:
                    from pathlib import Path
                    import sys
                    BASE = Path(__file__).resolve().parent.parent.parent.parent
                    sys.path.insert(0, str(BASE / "runtime"))
                    from semantic_map import update_map, feedback
                    update_map(cell.task, cell.output, weight=1.0)
                    feedback(cell.task, cell.output, correct=True)
                except Exception:
                    pass
            # Index
            self.index.add(cell.id, {"task": cell.task, "output": cell.output})
            # Roundabouts with memory offload
            self.roundabouts.route_with_memory(cell.id, {"output": cell.output, "task": cell.task})
            self.roundabouts.control.route_decision("task_complete", {"cell_id": cell.id})
            results.append(cell)
        return {"cells": results, "summary": f"Executed {len(results)} cells"}
