#!/usr/bin/env python3
"""
Redis Streams producer stub for music notes multi-worker safety
If redis is available, publish to stream. Otherwise write to local file.
"""
import json, time
from pathlib import Path
BASE = Path(__file__).resolve().parents[1]
SONG_LOG = BASE / "runtime" / "logs" / "song_notes.jsonl"
STREAM_LOG = BASE / "runtime" / "logs" / "music_stream.redis.jsonl"

def publish_note(note):
    # Try redis
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)
        r.xadd('music_notes', {'data': json.dumps(note)})
        return
    except Exception:
        pass
    # Fallback local
    STREAM_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(STREAM_LOG,'a',encoding='utf-8') as f:
        f.write(json.dumps({'ts': time.time(), 'note': note})+'\n')

def replay_stream(start_ts=None):
    # Replay from song_notes
    if not SONG_LOG.exists():
        return []
    notes = []
    with open(SONG_LOG,'r',encoding='utf-8') as f:
        for line in f:
            try:
                n = json.loads(line)
                if start_ts is None or n.get('ts',0) >= start_ts:
                    notes.append(n)
            except Exception:
                pass
    return notes

if __name__ == '__main__':
    # Demo: publish last 5 notes
    notes = replay_stream()
    for n in notes[-5:]:
        publish_note(n)
    print(f"Published {len(notes[-5:])} notes")
