"""
Graph-aware explanation composer
Traverses Wikidata graph and synthesizes explanations
"""
import sqlite3
from pathlib import Path
BASE = Path(__file__).resolve().parent.parent
DB_PATH = BASE / 'runtime' / 'logs' / 'wikidata_graph.db'

def get_label(conn, eid):
    cur = conn.cursor()
    cur.execute("SELECT value FROM labels WHERE entity_id=? AND lang='en' LIMIT 1", (eid,))
    row = cur.fetchone()
    return row[0] if row else eid

def get_entity_by_label(label):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT entity_id FROM labels WHERE value=? AND lang='en' LIMIT 1", (label,))
    row = cur.fetchone()
    conn.close()
    return row[0] if row else None

def explain(topic):
    eid = get_entity_by_label(topic)
    if not eid:
        return f"{topic} is not in your knowledge graph yet."
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT value FROM labels WHERE entity_id=? AND lang='en' LIMIT 1", (eid,))
    row = cur.fetchone()
    if not row:
        conn.close()
        return f"{topic} is not in your knowledge graph yet."
    label = row[0]
    # Try to fetch claims, handle missing table
    try:
        cur.execute("SELECT prop, value FROM claims WHERE entity_id=? LIMIT 10", (eid,))
        claims = cur.fetchall()
    except sqlite3.OperationalError:
        claims = []
    parts = [f"{label} is an entity in your knowledge graph."]
    for prop, val in claims:
        # Resolve if val is entity id
        if val.startswith('Q'):
            vlabel = get_label(conn, val)
            parts.append(f"It has {prop} = {vlabel}.")
        else:
            parts.append(f"It has {prop} = {val}.")
    conn.close()
    return " ".join(parts)

if __name__ == '__main__':
    import sys
    print(explain(sys.argv[1] if len(sys.argv)>1 else "France"))
