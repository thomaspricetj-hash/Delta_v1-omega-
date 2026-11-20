"""
Master Map Integrator
Feeds all graphs, indexes, ledgers, logs, notes into multilayer knowledge graph
"""
import sys, json, time
from pathlib import Path
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "runtime" / "master_map"))
from multilayer_graph import MultilayerGraph

graph = MultilayerGraph()

def load_json(p):
    try:
        return json.loads(Path(p).read_text())
    except:
        return {}

def log_ledger_entry(entry_type, payload):
    ledger_path = BASE / "runtime" / "logs" / "master_ledger.jsonl"
    entry = {
        "timestamp": time.time(),
        "type": entry_type,
        "payload": payload
    }
    try:
        sys.path.insert(0, str(BASE.parent.parent / "TRANSLATION_OS_V1" / "scripts"))
        from bitdrop_persistence import save_line
        save_line(ledger_path, entry)
    except Exception:
        try:
            with open(ledger_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except:
            pass

def integrate_all():
    # Indexes
    index_data = load_json(BASE / "infrastructure" / "index" / "index_v0.json")
    # Ledgers
    ledger_data = load_json(BASE / "scoreboard" / "capability_scoreboard.json")
    # Logs
    logs_data = {"log": []}
    # Notes
    notes_data = load_json(BASE / "workspace_memory" / "diary.json")
    # Ensure notes is a dict for MultilayerGraph.update
    if isinstance(notes_data, list):
        notes_data = {str(i): entry for i, entry in enumerate(notes_data)}
    
    graph.ingest("indexes", {"index": index_data, "nodes": {}, "edges": []})
    graph.ingest("ledgers", {"ledger": ledger_data if isinstance(ledger_data, list) else [ledger_data], "nodes": {}, "edges": []})
    graph.ingest("logs", {"log": logs_data, "nodes": {}, "edges": []})
    graph.ingest("notes", {"notes": notes_data, "nodes": {}, "edges": []})
    
    master = graph.fold()
    return master

if __name__ == '__main__':
    master = integrate_all()
    print(f"Integrated master map with {len(master['layers'])} layers, {len(master['indexes'])} indexes")
