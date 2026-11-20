"""
Byte-offset resumable Wikidata learning loop with property teaching
Teaches factual triples from Wikidata claims through full pipeline
"""
import sys, bz2, json, sqlite3, time, random
from pathlib import Path
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "orchestrator"))
from master_orchestrator_v0 import teach_full
sys.path.insert(0, str(BASE / "infrastructure" / "wikidata_qa"))
from wikidata_qa_generator import generate_qa

WIKIDATA_BZ2 = Path('C:/Users/thoma/Downloads/latest-all.json.bz2')
DB_PATH = BASE / 'runtime' / 'logs' / 'wikidata_graph.db'
STATE_PATH = BASE / 'runtime' / 'logs' / 'wikidata_learn_byteoffset_state.json'

# Property whitelist -> natural language template
PROP_MAP = {
    'P36': 'What is the capital of {subject}?',
    'P17': 'What country is {subject} in?',
    'P31': 'What is {subject} instance of?',
    'P1082': 'What is the population of {subject}?',
    'P571': 'When was {subject} founded?',
    'P625': 'Where is {subject} located?',
    'P279': 'What is {subject} subclass of?',
    'P131': 'What is {subject} located in?',
    'P19': 'Where was {subject} born?',
    'P20': 'Where did {subject} die?',
    'P102': 'What position did {subject} hold?',
    'P108': 'Where did {subject} work?',
    'P69': 'Where did {subject} study?',
    'P166': 'What award did {subject} receive?',
    'P57': 'Who directed {subject}?',
    'P50': 'Who wrote {subject}?',
    'P123': 'Who published {subject}?',
}

def load_state():
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text())
        except:
            pass
    return {"byte_offset": 0, "processed": 0}

def save_state(state):
    STATE_PATH.write_text(json.dumps(state))

def init_db():
    for i in range(5):
        try:
            conn = sqlite3.connect(DB_PATH, timeout=30)
            conn.execute('PRAGMA journal_mode=WAL')
            conn.execute('PRAGMA synchronous=NORMAL')
            return conn
        except sqlite3.OperationalError as e:
            if 'locked' in str(e).lower():
                time.sleep(2**i)
                continue
            raise
    raise RuntimeError("DB locked")

def get_label(conn, eid):
    cur = conn.cursor()
    cur.execute("SELECT value FROM labels WHERE entity_id=? AND lang='en' LIMIT 1", (eid,))
    row = cur.fetchone()
    return row[0] if row else None

def process_batch(byte_offset, limit=50000):
    import ijson
    # Bloom filter disabled due to size issues
    # sys.path.insert(0, str(BASE / "infrastructure" / "bloom_filter"))
    # from bloom import get_bloom
    bloom = None
    conn = init_db()
    cur = conn.cursor()
    cur.execute('CREATE TABLE IF NOT EXISTS entities (id TEXT PRIMARY KEY, type TEXT)')
    cur.execute('CREATE TABLE IF NOT EXISTS labels (entity_id TEXT, lang TEXT, value TEXT)')
    cur.execute('CREATE TABLE IF NOT EXISTS claims (entity_id TEXT, prop TEXT, value TEXT)')
    
    processed = 0
    start_time = time.time()
    last_report = start_time
    with bz2.open(WIKIDATA_BZ2, 'rb') as f:
        # bz2 seek is unreliable, start from beginning and rely on DB dedup
        # if byte_offset > 0: f.seek(byte_offset)
        parser = ijson.items(f, 'item')
        for item in parser:
            eid = item.get('id')
            if not eid:
                continue
            # Bloom filter disabled
            # quick existence check
            try:
                cur.execute('SELECT 1 FROM entities WHERE id=?', (eid,))
                if cur.fetchone():
                    processed += 1
                    continue
            except sqlite3.OperationalError:
                time.sleep(0.5)
                continue
            # Insert entity
            cur.execute('INSERT OR IGNORE INTO entities(id,type) VALUES(?,?)',(eid,item.get('type','')))
            # Labels
            subject_label = None
            for lang, lab in item.get('labels',{}).items():
                val = lab.get('value','')
                if not val:
                    continue
                cur.execute('INSERT INTO labels(entity_id,lang,value) VALUES(?,?,?)',(eid,lang,val))
                if lang == 'en':
                    subject_label = val
            # Claims
            if subject_label:
                for prop, claims_list in item.get('claims',{}).items():
                    if prop not in PROP_MAP:
                        continue
                    for claim in claims_list:
                        mainsnak = claim.get('mainsnak',{})
                        datavalue = mainsnak.get('datavalue',{})
                        value = datavalue.get('value')
                        if not value:
                            continue
                        # Resolve value
                        if isinstance(value, dict) and 'id' in value:
                            obj_id = value['id']
                            # store claim
                            cur.execute('INSERT INTO claims(entity_id,prop,value) VALUES(?,?,?)',(eid,prop,obj_id))
                            obj_label = get_label(conn, obj_id)
                            if obj_label:
                                # Generate multiple paraphrased Q/A for model-like training
                                for _ in range(3):
                                    q,a = generate_qa(subject_label, prop, obj_label)
                                    teach_full(q, a, reward=1.0)
                                # Negative example
                                q_neg,a_neg = generate_qa(subject_label, prop, obj_label, negative=True)
                                teach_full(q_neg, a_neg, reward=-1.0)
                        else:
                            # literal value
                            literal = str(value)
                            cur.execute('INSERT INTO claims(entity_id,prop,value) VALUES(?,?,?)',(eid,prop,literal))
                            for _ in range(3):
                                q,a = generate_qa(subject_label, prop, literal)
                                teach_full(q, a, reward=1.0)
                            q_neg,a_neg = generate_qa(subject_label, prop, literal, negative=True)
                            teach_full(q_neg, a_neg, reward=-1.0)
            processed += 1
            now = time.time()
            if now - last_report >= 60:
                elapsed = now - start_time
                print(f"[Progress] Processed {processed} entities in {elapsed:.1f}s")
                last_report = now
            if processed % 500 == 0:
                conn.commit()
                try:
                    state_offset = f.tell()
                except:
                    state_offset = byte_offset
                save_state({"byte_offset": state_offset, "processed": processed})
                print(f"Processed {processed} new entities, byte_offset={state_offset}")
            if processed >= limit:
                break
        conn.commit()
    conn.close()
    return processed

if __name__ == '__main__':
    while True:
        st = load_state()
        off = st.get('byte_offset',0)
        print(f"Resuming from byte offset {off}")
        try:
            n = process_batch(off, limit=50000)
            print(f"Batch taught {n} new entities")
        except Exception as e:
            import traceback
            print(f"Error: {e}, retrying in 10s")
            traceback.print_exc()
            time.sleep(10)
            continue
        time.sleep(60)
