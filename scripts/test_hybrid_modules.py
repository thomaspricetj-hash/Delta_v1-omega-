"""
Test all hybrid activation modules listed in master_orchestrator_v0.py
"""
import sys, json
from pathlib import Path
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "orchestrator"))

modules = [
    ("infrastructure/cognitive_gyrotron/cognitive_gyrotron", "cognitive_gyrotron"),
    ("infrastructure/north_star", "north_star"),
    ("infrastructure/gps/gps", "gps"),
    ("infrastructure/code_exec/code_executor", "code_exec"),
    ("infrastructure/decision_tree/decision_tree_engine", "decision_tree"),
    ("infrastructure/fiber_harmony/fiber_mapper", "fiber_harmony"),
    ("infrastructure/experience_epochs/experience_epochs", "experience_epochs"),
    ("infrastructure/meta_cognition/meta_cognition", "meta_cognition"),
    ("infrastructure/learning_loop/teach_workspace_memory", "learning_loop"),
    ("infrastructure/reasoning_scoreboard/reasoning_scoreboard_engine", "reasoning_scoreboard"),
    ("infrastructure/knowledge_grid/knowledge_grid_v0", "knowledge_grid"),
    ("infrastructure/fact_kb/fact_kb_lookup", "fact_kb"),
    ("infrastructure/true_hybrid/weave_transformer", "true_hybrid"),
    ("infrastructure/dax_memory_engine/memory_system", "dax_memory_engine"),
    ("infrastructure/training/weaveformer/weaveformer_converter", "training_weaveformer"),
    ("infrastructure/syntheticmind", "syntheticmind"),
    ("infrastructure/pattern_classifier/pattern_classifier", "pattern_classifier"),
    ("infrastructure/hierarchical_graph_attention/hierarchical_graph_attention", "hierarchical_graph_attention"),
    ("infrastructure/fractal_router/fractal_router", "fractal_router"),
    ("infrastructure/reasoning_chain/reasoning_chain", "reasoning_chain"),
    ("infrastructure/pattern_store/pattern_store", "pattern_store"),
    ("infrastructure/memory_organizer/memory_organizer", "memory_organizer"),
    ("infrastructure/photo_webbed_memory/photo_webbed_memory", "photo_webbed_memory"),
    ("infrastructure/picture_memory/picture_memory", "picture_memory"),
    ("infrastructure/attention/attention", "attention"),
    ("infrastructure/common_index/common_index", "common_index"),
    ("infrastructure/helpers/helpers", "helpers"),
    ("infrastructure/kv_cache_opt/kv_cache_opt", "kv_cache_opt"),
    ("infrastructure/learning_modalities/learning_modalities", "learning_modalities"),
    ("infrastructure/reasoning_graph/reasoning_graph", "reasoning_graph"),
    ("infrastructure/dax_kv_delta/dax_kv_delta", "dax_kv_delta"),
    ("infrastructure/helper_mesh/helper_mesh", "helper_mesh"),
    ("infrastructure/bitdrop_compression/bitdrop_compression", "bitdrop_compression"),
    ("infrastructure/computation_cache/computation_cache", "computation_cache"),
    ("infrastructure/pattern_recognizer/pattern_recognizer", "pattern_recognizer"),
    ("infrastructure/memory_attention/memory_attention", "memory_attention"),
    ("infrastructure/reasoning_heatmap/reasoning_heatmap", "reasoning_heatmap"),
    ("infrastructure/cognitive_maps/cognitive_maps", "cognitive_maps"),
    ("infrastructure/snapshot_memory/snapshot_memory", "snapshot_memory"),
]

results = []
for mod_path, name in modules:
    try:
        full_path = BASE / "infrastructure" / mod_path
        sys.path.insert(0, str(BASE / "infrastructure"))
        __import__(mod_path)
        results.append({"name": name, "status": "OK"})
    except Exception as e:
        results.append({"name": name, "status": "FAIL", "error": str(e)})

print(json.dumps(results, indent=2))
ok = sum(1 for r in results if r["status"]=="OK")
print(f"\nTested {len(results)} modules: {ok} OK, {len(results)-ok} FAIL")
