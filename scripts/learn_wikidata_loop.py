"""
Continuous Wikidata learning loop for Translation OS v1
Ingests Wikidata entities and teaches them via teach_full without runtime dependency.
Resumes from last processed offset to avoid re-teaching.
"""
import sys, bz2, json, sqlite3, time, os
from pathlib import Path
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "orchestrator"))
from master_orchestrator_v0 import teach_full

WIKIDATA_BZ2 = Path('C:/Users/thoma/Downloads/latest-all.json.bz2')
DB_PATH = BASE / 'runtime' / 'logs' / 'wikidata_graph.db'
STATE_PATH = BASE / 'runtime' / 'logs' / 'wikidata_learn_state.json'

def load_state():
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text())
        except:
            pass
    return {"processed_count": 0, "last_eid": None}

def save_state(state):
    STATE_PATH.write_text(json.dumps(state))

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute('PRAGMA journal_mode=WAL')
    conn.execute('PRAGMA synchronous=NORMAL')
    return conn

def process_stream(batch_size=500):
    import ijson
    state = load_state()
    start_count = state.get("processed_count", 0)
    conn = init_db()
    cur = conn.cursor()
    cur.execute('CREATE TABLE IF NOT EXISTS entities (id TEXT PRIMARY KEY, type TEXT)')
    cur.execute('CREATE TABLE IF NOT EXISTS labels (entity_id TEXT, lang TEXT, value TEXT)')
    cur.execute('CREATE INDEX IF NOT EXISTS idx_labels_eid ON labels(entity_id)')
    
    count = 0
    skipped = start_count
    # We cannot seek in bz2 stream, so we skip by counting items
    with bz2.open(WIKIDATA_BZ2,'rb') as f:
        parser = ijson.items(f,'item')
        for item in parser:
            # Skip already processed items
            if count < start_count:
                count += 1
                continue
            eid = item.get('id')
            if not eid:
                continue
            # Quick existence check to avoid duplicate teaching
            cur.execute('SELECT 1 FROM entities WHERE id=?', (eid,))
            if cur.fetchone():
                # Already ingested, just count
                count += 1
                continue
            # Upsert entity
            cur.execute('INSERT OR IGNORE INTO entities(id,type) VALUES(?,?)',(eid,item.get('type','')))
            # Labels
            taught = False
            for lang, lab in item.get('labels',{}).items():
                val = lab.get('value','')
                if not val:
                    continue
                cur.execute('INSERT INTO labels(entity_id,lang,value) VALUES(?,?,?)',(eid,lang,val))
                if lang == 'en':
                    # Teach only if not already taught - check labels table
                    cur.execute('SELECT 1 FROM labels WHERE entity_id=? AND lang=? AND value=?',(eid,lang,val))
                    if not cur.fetchone():
                        # New label
                        query = f"What is {val}?"
                        teach_full(query, val)
                        teach_full(f"Entity id for {val}", eid)
                        taught = True
            count += 1
            if count % batch_size == 0:
                conn.commit()
                save_state({"processed_count": count, "last_eid": eid})
                print(f"Processed {count} entities, taught new labels for batch")
                time.sleep(0.01)
        conn.commit()
        save_state({"processed_count": count, "last_eid": eid})
    conn.close()
    print(f"Learning complete, total processed {count}")

if __name__ == '__main__':
    while True:
        print(f"Starting learning batch from offset {load_state().get('processed_count',0)}")
        process_stream(batch_size=500)
        print("Batch done, sleeping 60s before next check...")
        time.sleep(60)
