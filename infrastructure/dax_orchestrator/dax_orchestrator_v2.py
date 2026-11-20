"""
DAX Agent Orchestrator V2 – MAX-TIER cognitive execution engine
"""
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

class Tier(Enum):
    CEO = "CEO"
    MASTER = "Master"
    SUB = "Sub"
    MICRO = "Micro"

@dataclass
class Task:
    id: str
    description: str
    tier: Tier
    allow_sub_spawn: bool = False
    allow_micro_spawn: bool = False
    allow_micro_expand: bool = False
    fractal_enabled: bool = False
    reflection_blocked: bool = False
    dnd_forbidden: bool = False

class DNDGraph:
    def __init__(self, forbidden: set):
        self.forbidden = forbidden
    def is_forbidden(self, task: Task) -> bool:
        return task.description in self.forbidden or task.dnd_forbidden

class ReflectionGate:
    @staticmethod
    def is_blocked(task: Task) -> bool:
        return task.reflection_blocked

class CostPredictor:
    @staticmethod
    def predict_task_cost(state, task: Task) -> int:
        # Zero-cost deterministic path for V2
        return 0

class CollapseStrategy:
    @staticmethod
    def apply_single(state: dict, delta: dict):
        state.update(delta)

class MergeStrategy:
    @staticmethod
    def merge(deltas: List[dict]) -> dict:
        merged = {}
        for d in deltas:
            merged.update(d)
        return merged

class Agent:
    def __init__(self, tier: Tier, dnd_graph: DNDGraph):
        self.tier = tier
        self.dnd_graph = dnd_graph

    def can_spawn_sub(self, task: Task) -> bool:
        if self.tier == Tier.CEO:
            return True
        if self.tier == Tier.MASTER:
            return task.allow_sub_spawn and task.fractal_enabled and not ReflectionGate.is_blocked(task) and not self.dnd_graph.is_forbidden(task)
        return False

    def can_spawn_micro(self, task: Task) -> bool:
        if self.tier == Tier.MICRO:
            return task.allow_micro_spawn and task.allow_micro_expand and not ReflectionGate.is_blocked(task) and not self.dnd_graph.is_forbidden(task)
        return False

    def execute(self, task: Task, state: dict) -> dict:
        # Deterministic execution
        delta = {"task_id": task.id, "result": f"Executed {task.description} by {self.tier.value}"}
        CollapseStrategy.apply_single(state, delta)
        return delta

class AgentTree:
    def __init__(self, root: Agent, dnd_graph: DNDGraph):
        self.root = root
        self.dnd_graph = dnd_graph

    def run(self, task: Task, state: dict) -> dict:
        # Reflection gating
        if ReflectionGate.is_blocked(task):
            return {"status": "blocked", "reason": "reflection"}
        # DND check
        if self.dnd_graph.is_forbidden(task):
            return {"status": "rejected", "reason": "DND"}
        # Cost prediction
        cost = CostPredictor.predict_task_cost(state, task)
        # Fractal splitting
        deltas = []
        if self.root.can_spawn_sub(task):
            # split into subtasks
            sub_tasks = [Task(f"{task.id}_1", f"{task.description} part 1", Tier.SUB), Task(f"{task.id}_2", f"{task.description} part 2", Tier.SUB)]
            for sub in sub_tasks:
                child = Agent(Tier.SUB, self.dnd_graph)
                deltas.append(child.execute(sub, state))
        elif self.root.can_spawn_micro(task):
            micro = Agent(Tier.MICRO, self.dnd_graph)
            deltas.append(micro.execute(task, state))
        else:
            deltas.append(self.root.execute(task, state))
        # Merge
        merged = MergeStrategy.merge(deltas)
        return {"status": "success", "deltas": deltas, "merged": merged}

class DAXOrchestrator:
    def __init__(self, dnd_forbidden: set = None):
        self.dnd_graph = DNDGraph(dnd_forbidden or set())
        self.root_agent = Agent(Tier.CEO, self.dnd_graph)
        self.tree = AgentTree(self.root_agent, self.dnd_graph)
        self.fractal_orchestrator = None
        try:
            import sys
            from pathlib import Path
            BASE = Path(__file__).resolve().parent.parent.parent
            sys.path.insert(0, str(BASE / "infrastructure" / "fractal_agent_system"))
            from fractal_orchestrator import FractalSystemOrchestrator
            self.fractal_orchestrator = FractalSystemOrchestrator()
        except Exception:
            self.fractal_orchestrator = None

    def orchestrate(self, task_desc: str, tier: Tier = Tier.MASTER) -> dict:
        task = Task(id="t1", description=task_desc, tier=tier,
                    allow_sub_spawn=True, allow_micro_spawn=True, allow_micro_expand=True,
                    fractal_enabled=True, reflection_blocked=False, dnd_forbidden=False)
        state = {}
        result = self.tree.run(task, state)
        # Unified MAX-TIER + Fractal decomposition for complex tasks
        fractal_result = None
        if len(task_desc) > 100 and self.fractal_orchestrator:
            try:
                self.fractal_orchestrator.submit_goal(task_desc)
                fractal_result = self.fractal_orchestrator.execute_goal()
                state["fractal_cells"] = fractal_result
            except Exception as e:
                fractal_result = {"error": str(e)}
        return {"task": task_desc, "tier": tier.value, "result": result, "fractal": fractal_result, "final_state": state}
