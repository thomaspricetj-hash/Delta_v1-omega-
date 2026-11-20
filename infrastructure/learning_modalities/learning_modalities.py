"""
Learning Modalities Engine – integrates 12 human learning modes into Translation OS
with cross-connected grid so learning happens on all levels, not one-way.
1 Learning Through Experience
2 Learning Through Observation
3 Learning Through Instruction
4 Learning Through Trial and Error
5 Memorization
6 Pattern Recognition
7 Learning Through Understanding
8 Learning Through Feedback
9 Learning Through Exploration
10 Learning Through Reflection
11 Learning Through Simulation
12 Learning Through Connection
"""
import json, time
from pathlib import Path
BASE = Path(__file__).resolve().parent.parent.parent
RUNTIME = BASE / "runtime" / "logs"
MODALITY_LOG = RUNTIME / "learning_modalities.jsonl"

# Cross-connected grid: each modality activates related modalities
MODALITY_GRID = {
    "experience": ["trial_error", "reflection", "feedback"],
    "observation": ["pattern_recognition", "connection", "memorization"],
    "instruction": ["understanding", "memorization", "connection"],
    "trial_error": ["experience", "reflection", "feedback"],
    "memorization": ["pattern_recognition", "connection", "reflection"],
    "pattern_recognition": ["understanding", "connection", "exploration"],
    "understanding": ["connection", "reflection", "simulation"],
    "feedback": ["reflection", "trial_error", "experience"],
    "exploration": ["observation", "pattern_recognition", "simulation"],
    "reflection": ["understanding", "feedback", "connection"],
    "simulation": ["understanding", "trial_error", "exploration"],
    "connection": ["understanding", "pattern_recognition", "memorization"]
}

def _save_log_entry(entry):
    try:
        import sys
        sys.path.insert(0, str(BASE / "scripts"))
        from bitdrop_persistence import save_line
        # Compress payload with BitDrop v2 to save space
        try:
            sys.path.insert(0, str(BASE.parent / "infrastructure"))
            from bitdrop_v2_integration import compress_data
            payload = json.dumps(entry).encode('utf-8')
            comp_info = compress_data(payload)
            # Store compression metadata alongside entry
            entry['_compressed'] = {
                "ratio": comp_info.get("ratio"),
                "original_len": comp_info.get("original_len"),
                "compressed_len": comp_info.get("compressed_len")
            }
        except Exception:
            pass
        save_line(MODALITY_LOG, entry)
    except Exception:
        pass

def log_modality(modality, query, response, meta=None):
    entry = {
        "ts": time.time(),
        "modality": modality,
        "query": query,
        "response": response,
        "meta": meta or {},
        "grid_activations": MODALITY_GRID.get(modality, [])
    }
    _save_log_entry(entry)

def log_weave(modality, query, response, meta=None):
    entry = {
        "ts": time.time(),
        "modality": modality,
        "query": query,
        "response": response,
        "meta": meta or {},
        "grid_activations": MODALITY_GRID.get(modality, [])
    }
    # BitDrop persistence
    try:
        import sys
        sys.path.insert(0, str(BASE / "scripts"))
        from bitdrop_persistence import save_line
        save_line(MODALITY_LOG, entry)
        # Cross-activate grid neighbors
        for neighbor in MODALITY_GRID.get(modality, []):
            neighbor_entry = dict(entry)
            neighbor_entry["modality"] = neighbor
            neighbor_entry["meta"] = {"triggered_by": modality}
            save_line(MODALITY_LOG, neighbor_entry)
    except:
        pass

# 1 Experience
def learn_by_experience(query, response, success=True):
    log_weave("experience", query, response, {"success": success})
    # reinforce via teach_full
    return "experience"

# 2 Observation
def learn_by_observation(query, observed_answer):
    log_weave("observation", query, observed_answer)
    return "observation"

# 3 Instruction
def learn_by_instruction(query, instruction_text):
    log_weave("instruction", query, instruction_text)
    return "instruction"

# 4 Trial and Error
def learn_trial_error(query, attempts, best_idx):
    log_weave("trial_error", query, attempts[best_idx], {"attempts": attempts, "best_idx": best_idx})
    return "trial_error"

# 5 Memorization
def memorize(query, response, reps=3):
    for i in range(reps):
        log_weave("memorization", query, response, {"rep": i+1})
    return "memorization"

# 6 Pattern Recognition
def recognize_pattern(query, pattern_id):
    log_weave("pattern_recognition", query, "", {"pattern_id": pattern_id})
    return "pattern_recognition"

# 7 Understanding
def learn_understanding(query, principle):
    log_weave("understanding", query, principle)
    return "understanding"

# 8 Feedback
def learn_feedback(query, feedback):
    log_weave("feedback", query, feedback)
    return "feedback"

# 9 Exploration
def explore(query):
    log_weave("exploration", query, "")
    return "exploration"

# 10 Reflection
def reflect(query, past_errors):
    log_weave("reflection", query, "", {"past_errors": past_errors})
    return "reflection"

# 11 Simulation
def simulate(query, simulated_outcome):
    log_weave("simulation", query, simulated_outcome)
    return "simulation"

# 12 Connection
def connect_knowledge(query, linked_concepts):
    log_weave("connection", query, "", {"links": linked_concepts})
    return "connection"

def route_modality(query, diagnostics):
    # Simple heuristic routing
    ql = query.lower()
    if "my favorite" in ql:
        return "experience"
    if "explain why" in ql or "how does" in ql:
        return "understanding"
    if "what is" in ql and "?" in ql:
        return "instruction"
    if diagnostics.get("needs_rework"):
        return "trial_error"
    if diagnostics.get("evaluation", {}).get("quality_score", 0) < 0.6:
        return "reflection"
    return "pattern_recognition"

# Extended modalities requested
EXTENDED_MODALITIES = {
    "reading": ["memorization", "pattern_recognition", "understanding"],
    "training_on_datasets": ["repetition", "multiple_epochs", "pattern_recognition"],
    "repetition": ["memorization", "training_on_datasets", "practice"],
    "multiple_epochs": ["training_on_datasets", "repetition", "reinforcement_learning"],
    "practice": ["experience", "trial_error", "task_execution"],
    "task_execution": ["practice", "goal_directed_agents", "project_learning"],
    "demonstration_learning": ["observation", "imitation", "understanding"],
    "reinforcement_learning": ["trial_error", "feedback", "multiple_epochs"],
    "questioning": ["reflection", "self_reflection_loops", "understanding"],
    "self_reflection_loops": ["reflection", "questioning", "error_analysis_feedback"],
    "feature_extraction": ["pattern_recognition", "attention_networks", "understanding"],
    "teaching_others": ["knowledge_distillation", "understanding", "reflection"],
    "knowledge_distillation": ["teaching_others", "compression", "understanding"],
    "discussion": ["multi_agent_reasoning", "questioning", "understanding"],
    "multi_agent_reasoning": ["discussion", "goal_directed_agents", "internal_world_models"],
    "visualization": ["pattern_recognition", "understanding", "internal_world_models"],
    "internal_world_models": ["visualization", "simulation", "multi_agent_reasoning"],
    "chunking": ["hierarchical_memory", "pattern_recognition", "feature_extraction"],
    "hierarchical_memory": ["chunking", "memorization", "connection"],
    "association": ["connection", "pattern_recognition", "hierarchical_memory"],
    "attention_networks": ["feature_extraction", "pattern_recognition", "understanding"],
    "error_analysis_feedback": ["feedback", "self_reflection_loops", "trial_error"],
    "project_learning": ["task_execution", "goal_directed_agents", "practice"],
    "goal_directed_agents": ["project_learning", "task_execution", "multi_agent_reasoning"],
    "exploration": ["curiosity_driven_learning", "observation", "pattern_recognition"],
    "curiosity_driven_learning": ["exploration", "questioning", "internal_world_models"]
}

# Merge extended into main grid for cross-activation
MODALITY_GRID.update(EXTENDED_MODALITIES)

# Weave Lattice – cross-connecting grids for emergent interplay
WEAVE_LATTICE = {
    # Core learning weave
    "reading": ["training_on_datasets", "repetition", "visualization"],
    "training_on_datasets": ["multiple_epochs", "repetition", "reinforcement_learning"],
    "repetition": ["memorization", "practice", "multiple_epochs"],
    "multiple_epochs": ["training_on_datasets", "reinforcement_learning", "knowledge_distillation"],
    "practice": ["task_execution", "trial_error", "experience"],
    "task_execution": ["project_learning", "goal_directed_agents", "practice"],
    "demonstration_learning": ["observation", "imitation", "multi_agent_reasoning"],
    "reinforcement_learning": ["trial_error", "feedback", "curiosity_driven_learning"],
    "questioning": ["self_reflection_loops", "discussion", "internal_world_models"],
    "self_reflection_loops": ["reflection", "error_analysis_feedback", "questioning"],
    "feature_extraction": ["attention_networks", "chunking", "pattern_recognition"],
    "teaching_others": ["knowledge_distillation", "discussion", "reflection"],
    "knowledge_distillation": ["teaching_others", "compression", "internal_world_models"],
    "discussion": ["multi_agent_reasoning", "questioning", "teaching_others"],
    "multi_agent_reasoning": ["goal_directed_agents", "internal_world_models", "discussion"],
    "visualization": ["internal_world_models", "pattern_recognition", "reading"],
    "internal_world_models": ["simulation", "visualization", "curiosity_driven_learning"],
    "chunking": ["hierarchical_memory", "feature_extraction", "association"],
    "hierarchical_memory": ["chunking", "memorization", "connection"],
    "association": ["connection", "hierarchical_memory", "pattern_recognition"],
    "attention_networks": ["feature_extraction", "pattern_recognition", "understanding"],
    "error_analysis_feedback": ["feedback", "self_reflection_loops", "trial_error"],
    "project_learning": ["goal_directed_agents", "task_execution", "practice"],
    "goal_directed_agents": ["project_learning", "multi_agent_reasoning", "exploration"],
    "curiosity_driven_learning": ["exploration", "questioning", "internal_world_models"],
    # Weave back to original 12
    "experience": ["practice", "trial_error", "reflection"],
    "observation": ["demonstration_learning", "visualization", "pattern_recognition"],
    "instruction": ["reading", "teaching_others", "understanding"],
    "trial_error": ["reinforcement_learning", "practice", "error_analysis_feedback"],
    "memorization": ["repetition", "hierarchical_memory", "reading"],
    "pattern_recognition": ["feature_extraction", "association", "attention_networks"],
    "understanding": ["questioning", "internal_world_models", "reflection"],
    "feedback": ["error_analysis_feedback", "reinforcement_learning", "reflection"],
    "exploration": ["curiosity_driven_learning", "goal_directed_agents", "observation"],
    "reflection": ["self_reflection_loops", "questioning", "feedback"],
    "simulation": ["internal_world_models", "visualization", "understanding"],
    "connection": ["association", "hierarchical_memory", "understanding"]
}

def log_weave(modality, query, response, meta=None, depth=None, layer=0, visited=None):
    if visited is None:
        visited = set()
    if modality in visited:
        return
    visited.add(modality)
    # Auto-expand depth based on layer and system maturity
    MAX_DEPTH = 7
    BASE_DEPTH = 3
    if depth is None:
        # Auto-expand: deeper as we go, capped
        depth = min(MAX_DEPTH, BASE_DEPTH + layer)
    # Log primary modality with layer info
    log_meta = dict(meta or {})
    log_meta.update({"weave_layer": layer, "weave_depth_remaining": depth})
    log_modality(modality, query, response, log_meta)
    # Cross-connecting grids: log explicit grid edges for this modality
    try:
        import sys
        sys.path.insert(0, str(BASE / "scripts"))
        from bitdrop_persistence import save_line
        for grid_neighbor in MODALITY_GRID.get(modality, []):
            grid_entry = {
                "ts": time.time(),
                "modality": modality,
                "grid_neighbor": grid_neighbor,
                "query": query,
                "response": response,
                "meta": {"weave_from": modality, "weave_layer": layer, "grid_cross_connect": True, **(meta or {})},
                "weave_lattice": True,
                "weave_layer": layer
            }
            save_line(MODALITY_LOG, grid_entry)
    except:
        pass
    # Weave lattice cross-activation with expanding layers
    if depth > 0:
        for neighbor in WEAVE_LATTICE.get(modality, []):
            # Log neighbour via weave entry
            weave_entry = {
                "ts": time.time(),
                "modality": neighbor,
                "query": query,
                "response": response,
                "meta": {"weave_from": modality, "weave_layer": layer+1, "weave_depth_remaining": depth-1, **(meta or {})},
                "grid_activations": MODALITY_GRID.get(neighbor, []),
                "weave_lattice": True,
                "weave_layer": layer+1
            }
            try:
                import sys
                sys.path.insert(0, str(BASE / "scripts"))
                from bitdrop_persistence import save_line
                save_line(MODALITY_LOG, weave_entry)
            except:
                pass
            # Recurse to next layer – lattice expands
            log_weave(neighbor, query, response, meta, depth-1, layer+1, visited)

# Convenience wrappers for extended modalities
def learn_reading(query, text):
    log_weave("reading", query, text)
    return "reading"

def learn_training_on_datasets(dataset_id, epochs):
    log_weave("training_on_datasets", dataset_id, f"epochs={epochs}")
    return "training_on_datasets"

def learn_repetition(query, response, reps=3):
    for i in range(reps):
        log_weave("repetition", query, response, {"rep": i+1})
    return "repetition"

def learn_multiple_epochs(query, epochs):
    log_weave("multiple_epochs", query, f"epochs={epochs}")
    return "multiple_epochs"

def learn_practice(query, attempts):
    log_weave("practice", query, str(attempts))
    return "practice"

def learn_task_execution(task_id, result):
    log_weave("task_execution", task_id, result)
    return "task_execution"

def learn_demonstration(query, demo):
    log_weave("demonstration_learning", query, demo)
    return "demonstration_learning"

def learn_reinforcement(query, reward):
    log_weave("reinforcement_learning", query, f"reward={reward}")
    return "reinforcement_learning"

def learn_questioning(query, questions):
    log_weave("questioning", query, str(questions))
    return "questioning"

def learn_self_reflection(query, insights):
    log_weave("self_reflection_loops", query, insights)
    return "self_reflection_loops"

def learn_feature_extraction(query, features):
    log_weave("feature_extraction", query, str(features))
    return "feature_extraction"

def learn_teaching_others(query, explanation):
    log_weave("teaching_others", query, explanation)
    return "teaching_others"

def learn_knowledge_distillation(teacher_output, student_input):
    log_weave("knowledge_distillation", teacher_output, student_input)
    return "knowledge_distillation"

def learn_discussion(topic, participants):
    log_weave("discussion", topic, str(participants))
    return "discussion"

def learn_multi_agent_reasoning(scenario, agents):
    log_weave("multi_agent_reasoning", scenario, str(agents))
    return "multi_agent_reasoning"

def learn_visualization(query, viz_data):
    log_weave("visualization", query, viz_data)
    return "visualization"

def learn_internal_world_models(state, prediction):
    log_weave("internal_world_models", state, prediction)
    return "internal_world_models"

def learn_chunking(query, chunks):
    log_weave("chunking", query, str(chunks))
    return "chunking"

def learn_hierarchical_memory(query, levels):
    log_weave("hierarchical_memory", query, str(levels))
    return "hierarchical_memory"

def learn_association(query, pairs):
    log_weave("association", query, str(pairs))
    return "association"

def learn_attention_networks(query, attention_map):
    log_weave("attention_networks", query, str(attention_map))
    return "attention_networks"

def learn_error_analysis(query, errors):
    log_weave("error_analysis_feedback", query, str(errors))
    return "error_analysis_feedback"

def learn_project(query, project_id):
    log_weave("project_learning", query, project_id)
    return "project_learning"

def learn_goal_directed(query, goal):
    log_weave("goal_directed_agents", query, goal)
    return "goal_directed_agents"

def learn_curiosity(query, novelty):
    log_weave("curiosity_driven_learning", query, str(novelty))
    return "curiosity_driven_learning"

