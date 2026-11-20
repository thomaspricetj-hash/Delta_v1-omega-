"""
Project & Progress Planner
Keeps track of projects and progress without manual note review.
Maps out milestones beforehand to speed up progress.
"""
import json
from pathlib import Path
from datetime import datetime

BASE = Path(__file__).resolve().parent.parent
WORKSPACE_MEM = BASE / "workspace_memory"
RUNTIME_LOGS = BASE / "runtime" / "logs"
PLAN_PATH = RUNTIME_LOGS / "project_plan.json"
PROGRESS_PATH = RUNTIME_LOGS / "progress_tracker.json"

def load_json(p):
    try:
        return json.loads(Path(p).read_text())
    except:
        return None

def collect_projects():
    projects = {}
    # Project book
    proj_book = load_json(WORKSPACE_MEM / "project_book.json")
    if proj_book:
        projects["project_book"] = proj_book
    # Diary entries for active topics
    diary = load_json(WORKSPACE_MEM / "diary.json")
    if diary:
        projects["diary_entries"] = len(diary)
    # Future planner
    future = load_json(WORKSPACE_MEM / "future_book.json")
    if future:
        projects["future"] = future
    # Workspace memory loader summary
    return projects

def build_dependency_dag(future_book):
    # Auto-generate dependency DAG from future_book blocked goals
    goals = []
    if future_book:
        current = future_book.get("current_goal")
        if current:
            goals.append({"id":"current","name":current,"depends_on":[]})
        for bg in future_book.get("blocked_goals", []):
            gid = bg.get("id") or bg.get("goal","").lower().replace(" ","_")
            goals.append({
                "id": gid,
                "name": bg.get("goal",""),
                "depends_on": bg.get("depends_on", [])
            })
    # Build simple DAG edges
    edges = []
    for g in goals:
        for dep in g.get("depends_on", []):
            edges.append({"from": dep, "to": g["id"]})
    return {"goals": goals, "edges": edges}

def build_roadmap():
    projects = collect_projects()
    future = load_json(WORKSPACE_MEM / "future_book.json")
    dag = build_dependency_dag(future)
    roadmap = {
        "generated_at": datetime.utcnow().isoformat(),
        "projects": projects,
        "dependency_dag": dag,
        "milestones": [
            {"id": "wikidata_ingest", "name": "Wikidata continuous ingest", "status": "active", "progress": 0.6},
            {"id": "emergence_training", "name": "Emergence training 2h/day", "status": "active", "progress": 0.5},
            {"id": "dream_state", "name": "Dream state consolidation", "status": "active", "progress": 0.7},
            {"id": "master_map", "name": "Master multilayered foldable knowledge graph", "status": "in_progress", "progress": 0.4},
            {"id": "reasoning_workspace", "name": "Reasoning workspace integration", "status": "done", "progress": 1.0},
            {"id": "confidence_max3d", "name": "Confidence + MAX-3D pipeline", "status": "done", "progress": 1.0},
        ],
        "next_actions": [
            "Run emergence tests daily and log scores",
            "Continue Wikidata byte-offset teaching",
            "Review dream_state.log for new connections",
            "Fold indexes/ledgers/logs/notes into master map nightly"
        ]
    }
    return roadmap

def save_plan(roadmap):
    from bitdrop_persistence import save_json
    save_json(PLAN_PATH, roadmap)

def load_progress():
    # Try plain first, fallback to empty
    if PROGRESS_PATH.exists():
        try:
            return json.loads(PROGRESS_PATH.read_text())
        except:
            return {"completed_milestones": [], "last_update": None}
    return {"completed_milestones": [], "last_update": None}

def update_progress():
    from bitdrop_persistence import save_json
    roadmap = build_roadmap()
    prog = load_progress()
    # Simple heuristic progress update
    prog["last_update"] = datetime.utcnow().isoformat()
    prog["roadmap_summary"] = roadmap["milestones"]
    save_json(PROGRESS_PATH, prog)
    save_plan(roadmap)
    return roadmap, prog

if __name__ == "__main__":
    roadmap, prog = update_progress()
    print(json.dumps(roadmap, indent=2))
    print("\nProgress tracker updated.")
