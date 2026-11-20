"""
Graph-aware explanation composer for Translation OS v1
"""
import sys, sqlite3, json
from pathlib import Path
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "orchestrator"))
import master_orchestrator_v0 as m

DB_PATH = BASE / 'runtime' / 'logs' / 'wikidata_graph.db'
PREF_PATH = BASE / 'runtime' / 'logs' / 'user_preferences.json'

def load_prefs():
    if PREF_PATH.exists():
        try:
            return json.loads(PREF_PATH.read_text())
        except:
            pass
    return {}

def get_entity_by_label(label):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT entity_id FROM labels WHERE value=? AND lang='en' LIMIT 1", (label,))
    row = cur.fetchone()
    conn.close()
    return row[0] if row else None

def get_claims(eid):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT prop, value FROM claims WHERE entity_id=? LIMIT 10", (eid,))
    rows = cur.fetchall()
    conn.close()
    return rows

def explain(topic):
    # Try find entity
    eid = get_entity_by_label(topic)
    if not eid:
        # Try via orchestrator
        r = m.run_orchestrator(f"What is {topic}?")
        resp = r.get('response')
        if resp:
            return f"{topic} is {resp}. It is a concept in your knowledge graph."
        return f"{topic} is a concept. No facts taught yet."
    label = topic
    claims = get_claims(eid)
    if not claims:
        return f"{label} is an entity in your knowledge graph with no taught claims yet."
    # Build mini-graph explanation
    parts = [f"{label} is an entity in your knowledge graph."]
    for prop, val in claims[:5]:
        parts.append(f"It has property {prop} = {val}.")
    return " ".join(parts)

if __name__ == '__main__':
    print(explain(sys.argv[1] if len(sys.argv)>1 else "quantum entanglement"))
