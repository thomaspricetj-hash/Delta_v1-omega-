"""
Cognitive Organizer
Integrates 12 human-like organization structures into Translation OS
"""
import sys
from pathlib import Path
BASE = Path(__file__).resolve().parent.parent.parent

sys.path.insert(0, str(BASE / "infrastructure" / "foldable_map"))
from foldable_map import get_foldable_map

fm = get_foldable_map()

# 1. Hierarchies
def build_hierarchy(concept, children):
    fm.put(concept, {"type":"hierarchy","children":children}, layer="hierarchy", meta={"structure":"hierarchy"})

# 2. Categories
def categorize(items, category):
    fm.put(category, {"type":"category","items":items}, layer="category", meta={"structure":"category"})

# 3. Associations
def associate(concept, links):
    fm.put(concept, {"type":"association","links":links}, layer="association", meta={"structure":"association"})

# 4. Stories / Sequences
def store_sequence(name, steps):
    fm.put(name, {"type":"sequence","steps":steps}, layer="sequence", meta={"structure":"sequence"})

# 5. Spatial Maps
def store_spatial(map_name, locations):
    fm.put(map_name, {"type":"spatial","locations":locations}, layer="spatial", meta={"structure":"spatial"})

# 6. Patterns / Templates
def store_template(name, schema):
    fm.put(name, {"type":"template","schema":schema}, layer="template", meta={"structure":"template"})

# 7. Importance Ranking
def rank_items(items):
    # items: list of (item, score)
    fm.put("importance_rank", {"type":"ranking","items":items}, layer="ranking", meta={"structure":"ranking"})

# 8. Timelines
def store_timeline(name, events):
    fm.put(name, {"type":"timeline","events":events}, layer="timeline", meta={"structure":"timeline"})

# 9. Cause-Effect Chains
def store_causality(cause, effect_chain):
    fm.put(cause, {"type":"causality","chain":effect_chain}, layer="causality", meta={"structure":"causality"})

# 10. Abstraction Layers
def store_abstraction(concept, layers):
    fm.put(concept, {"type":"abstraction","layers":layers}, layer="abstraction", meta={"structure":"abstraction"})

# 11. Emotional Tagging
def tag_emotion(memory_key, emotion, intensity):
    fm.put(memory_key, {"type":"emotion","emotion":emotion,"intensity":intensity}, layer="emotion", meta={"structure":"emotion"})

# 12. Fractal Organization
def store_fractal(name, levels):
    fm.put(name, {"type":"fractal","levels":levels}, layer="fractal", meta={"structure":"fractal"})

def organize_query(query, answer, diagnostics):
    """
    Apply cognitive organization structures to a query-answer pair
    """
    # Example heuristics
    # Hierarchy: if answer contains 'is a' -> parent-child
    # Categories: group by topic
    # Associations: link query words
    # Stories: if query contains temporal words
    # etc.
    # For now, store basic association and category
    associate(query, {"answer": answer, "confidence": diagnostics.get("confidence_prior", 0.5)})
    # Store abstraction
    store_abstraction(query, ["concept","instance","detail"])
    # Emotional tagging based on confidence
    conf = diagnostics.get("confidence_prior", 0.5)
    emotion = "confident" if conf > 0.7 else "uncertain" if conf < 0.4 else "neutral"
    tag_emotion(query, emotion, conf)
    return True

__all__ = ["organize_query","build_hierarchy","categorize","associate","store_sequence","store_spatial","store_template","rank_items","store_timeline","store_causality","store_abstraction","tag_emotion","store_fractal"]
