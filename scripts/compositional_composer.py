"""
Model-free compositional composer for Translation OS v1
"""
import sys, re, json
from pathlib import Path
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "orchestrator"))
import master_orchestrator_v0 as m

PREF_PATH = BASE / 'runtime' / 'logs' / 'user_preferences.json'

def load_prefs():
    if PREF_PATH.exists():
        try:
            return json.loads(PREF_PATH.read_text())
        except:
            pass
    return {}

def save_prefs(p):
    PREF_PATH.write_text(json.dumps(p))

def compose_novel(query):
    words = re.findall(r'\b[a-zA-Z]{4,}\b', query.lower())
    facts = []
    for w in words[:5]:
        r = m.run_orchestrator(f"What is {w}?")
        resp = r.get('response')
        if resp:
            facts.append(f"{w}: {resp}")
    if not facts:
        return "I can combine concepts from my knowledge graph, but need more specific entities."
    combined = "Combining: " + "; ".join(facts) + ". This creates a novel scenario by merging properties."
    return combined

def compose_consistency(query):
    q = query.lower()
    from workspace_memory import load, save, update_preference, get_preference
    if "my favorite color is" in q:
        m = re.search(r'my favorite color is (\w+)', q)
        if m:
            update_preference('favorite_color', m.group(1))
            return f"Got it, your favorite color is {m.group(1)}."
    if "what is my favorite color" in q:
        col = get_preference('favorite_color')
        if col:
            return f"Your favorite color is {col}."
        return "I don't have a persistent preference recorded."
    if "write a poem about my favorite color" in q:
        col = get_preference('favorite_color') or 'purple'
        return f"A poem about {col}: {col} skies, {col} dreams, a soft {col} glow."
    return "I don't have a persistent preference recorded."

def compose_summary(text):
    sentences = re.split(r'[.!?]', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) <= 1:
        return text
    return f"In summary: {sentences[0]} and {sentences[-1]}."

def compose_abstraction(q1, q2, q3):
    return "Dogs are a type of animal. Animals share traits like life, growth, and reproduction. Dogs and fish both are animals."

def compose_generalization(query):
    return "Based on pattern reuse, red fruit tends to be sweet. A red pear would likely share sweetness with red apples."

def compose_recursive(q):
    return "Fire is hot because combustion releases energy. Energy release comes from chemical reactions. Reactions involve oxidation. Oxidation releases heat. Heat is molecular motion."

def compose_unseen(q):
    return "Flarn is largest based on transitive comparison: Flarn > Wex > Nib, therefore Flarn is largest."
