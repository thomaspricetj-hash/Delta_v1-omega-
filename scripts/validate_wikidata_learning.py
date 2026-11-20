"""
Validate that Wikidata teaching is working correctly
Samples recent entities and checks retrieval from memory
"""
import sys, sqlite3, random
from pathlib import Path
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "orchestrator"))
import master_orchestrator_v0 as m

DB_PATH = BASE / 'runtime' / 'logs' / 'wikidata_graph.db'

def get_recent_entities(conn, limit=50):
    cur = conn.cursor()
    cur.execute("""
        SELECT e.id, l.value FROM entities e
        JOIN labels l ON e.id = l.entity_id
        WHERE l.lang='en'
        ORDER BY e.id DESC
        LIMIT ?
    """, (limit,))
    return cur.fetchall()

def test_entity(eid, label):
    # Try a few property questions
    questions = [
        f"What is the capital of {label}?",
        f"What country is {label} in?",
        f"What is {label} instance of?",
    ]
    results = []
    for q in questions:
        r = m.run_orchestrator(q)
        resp = r.get('response')
        src = r.get('response_source')
        results.append((q, resp, src))
    return results

if __name__ == '__main__':
    conn = sqlite3.connect(DB_PATH)
    entities = get_recent_entities(conn, limit=20)
    print(f"Testing {len(entities)} recent entities")
    for eid, label in entities:
        print(f"\nEntity {eid} - {label}")
        res = test_entity(eid, label)
        for q, resp, src in res:
            print(f"  Q: {q}")
            print(f"  -> {resp} [{src}]")
    conn.close()
    print("\nValidation complete")
