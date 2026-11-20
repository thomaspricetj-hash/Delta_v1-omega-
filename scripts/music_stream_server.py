"""
Production-grade music telemetry streamer for Translation OS
- FastAPI + WebSocket live note stream
- Ring buffer for backpressure
- Prometheus metrics endpoint
- Health/readiness probes
"""
import asyncio
import json
import logging
from pathlib import Path
from collections import deque
from typing import Set

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, PlainTextResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("music_stream")

BASE = Path(__file__).resolve().parents[1]
SONG_LOG = BASE / "runtime" / "logs" / "song_notes.jsonl"
METRICS_FILE = BASE / "runtime" / "logs" / "music_metrics.prom"

app = FastAPI(title="Translation OS Music Stream")
active_ws: Set[WebSocket] = set()
recent_notes = deque(maxlen=1000)

def load_existing_notes():
    if not SONG_LOG.exists():
        return
    try:
        with open(SONG_LOG, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    recent_notes.append(json.loads(line))
                except Exception:
                    continue
    except Exception:
        pass

async def broadcaster():
    load_existing_notes()
    last_pos = SONG_LOG.stat().st_size if SONG_LOG.exists() else 0
    while True:
        try:
            if SONG_LOG.exists():
                size = SONG_LOG.stat().st_size
                if size > last_pos:
                    with open(SONG_LOG, "r", encoding="utf-8") as f:
                        f.seek(last_pos)
                        for line in f:
                            try:
                                note = json.loads(line)
                                recent_notes.append(note)
                                msg = json.dumps({"type": "note", "data": note})
                                dead = set()
                                for ws in active_ws:
                                    try:
                                        await ws.send_text(msg)
                                    except Exception:
                                        dead.add(ws)
                                for ws in dead:
                                    active_ws.discard(ws)
                            except Exception:
                                continue
                    last_pos = size
                elif size < last_pos:
                    last_pos = 0
        except Exception:
            pass
        await asyncio.sleep(0.05)

@app.get("/healthz")
def healthz():
    return {"status": "ok", "notes": len(recent_notes)}

@app.get("/readyz")
def readyz():
    return {"status": "ready"}

@app.get("/metrics")
def metrics():
    if METRICS_FILE.exists():
        return PlainTextResponse(METRICS_FILE.read_text(encoding="utf-8"))
    return PlainTextResponse("# no metrics yet\n")

@app.get("/api/harmony")
def harmony():
    try:
        from infrastructure.music_mapper.music_mapper import get_harmony_summary
        return JSONResponse(get_harmony_summary())
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.websocket("/ws/notes")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    active_ws.add(ws)
    for note in list(recent_notes)[-100:]:
        await ws.send_text(json.dumps({"type": "note", "data": note}))
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        active_ws.discard(ws)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(broadcaster())
    log.info("Music stream server starting")

if __name__ == "__main__":
    uvicorn.run("music_stream_server:app", host="0.0.0.0", port=8001, log_level="info")
