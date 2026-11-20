"""
Music Mapper – turn correct responses into notes
Each correct/ high-confidence response becomes a musical note.
Notes are chained into a song and harmony is derived from fiber scores.
"""
import json
from pathlib import Path
import hashlib
import numpy as np

BASE = Path(__file__).resolve().parents[2]
SONG_LOG = BASE / "runtime" / "logs" / "song_notes.jsonl"
MIDI_OUT = BASE / "runtime" / "logs" / "song.wav"

NOTE_BASE = 60  # C4
NOTE_RANGE = 25 # up to ~G5

def response_to_note(response, confidence=0.5, weight=0.5):
    # Deterministic note from response hash
    h = int(hashlib.sha256(response.encode()).hexdigest(), 16)
    base_note = NOTE_BASE + (h % NOTE_RANGE)
    # Weight influences pitch: pluck weight shifts note up/down
    # weight in [0,1] maps to -4..+4 semitones
    weight_offset = int((weight - 0.5) * 16)  # -8 to +8, then clamp
    weight_offset = max(-4, min(4, weight_offset))
    note = base_note + weight_offset
    # Duration from confidence
    duration = 0.3 + 0.7 * confidence
    return note, duration

INSTRUMENT_MAP = {
    "code_exec": "bass",
    "fc_mbfi_fact_core": "piano",
    "native_generator": "strings",
    "compositional:explanation_composer_v2": "strings",
    "reasoning": "brass",
    "native": "strings"
}

def get_instrument(source):
    return INSTRUMENT_MAP.get(source, "piano")

def log_note(query, response, confidence, source, weight=0.5, latency_ms=None, request_id=None):
    note, duration = response_to_note(response, confidence, weight)
    instrument = get_instrument(source)
    velocity = max(0.0, min(1.0, confidence))
    entry = {
        "ts": __import__('time').time(),
        "request_id": request_id,
        "query": query,
        "response": response[:200],
        "source": source,
        "instrument": instrument,
        "confidence": confidence,
        "weight": weight,
        "velocity": velocity,
        "latency_ms": latency_ms,
        "note": note,
        "duration": duration
    }
    SONG_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(SONG_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return entry

def note_to_freq(note):
    return 440.0 * 2 ** ((note - 69) / 12)

def render_song(output_path=MIDI_OUT, sr=44100):
    if not SONG_LOG.exists():
        return None
    notes = []
    with open(SONG_LOG, "r", encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            notes.append(e)
    if not notes:
        return None
    # Build waveform with per-instrument timbre and velocity
    wav = np.array([], dtype=np.float32)
    for e in notes:
        freq = note_to_freq(e["note"])
        dur = e.get("duration", 0.5)
        velocity = float(e.get("velocity", 0.5))
        instrument = e.get("instrument", "piano")
        t = np.linspace(0, dur, int(sr * dur), False)
        # Simple timbre per instrument
        if instrument == "bass":
            wave = 0.3 * velocity * (np.sin(2 * np.pi * freq * t) + 0.5 * np.sin(2 * np.pi * 2 * freq * t))
        elif instrument == "brass":
            wave = 0.3 * velocity * (np.sin(2 * np.pi * freq * t) + 0.3 * np.sin(2 * np.pi * 3 * freq * t))
        elif instrument == "strings":
            wave = 0.3 * velocity * np.sin(2 * np.pi * freq * t)
        else:  # piano
            wave = 0.3 * velocity * np.sin(2 * np.pi * freq * t)
        # Simple fade in/out
        fade_len = int(sr * 0.01)
        if len(wave) > fade_len * 2:
            fade = np.linspace(0,1, fade_len)
            wave[:fade_len] *= fade
            wave[-fade_len:] *= fade[::-1]
        wav = np.concatenate([wav, wave])
    # Normalize
    if wav.size > 0:
        wav = wav / np.max(np.abs(wav) + 1e-9)
    # Save as wav
    try:
        import scipy.io.wavfile as wavfile
        wavfile.write(output_path, sr, (wav * 32767).astype(np.int16))
    except Exception:
        # Fallback to raw numpy
        np.save(output_path.with_suffix('.npy'), wav)
    return output_path

def interval_consonance(i):
    # Simple consonance map for semitone intervals mod 12
    consonant = {0:1.0, 3:0.9, 4:0.9, 5:0.8, 7:0.85, 8:0.8, 9:0.7, 10:0.6, 11:0.5, 12:1.0}
    return consonant.get(i % 12, 0.5)

def export_prometheus_metrics(path=None):
    if path is None:
        path = BASE / "runtime" / "logs" / "music_metrics.prom"
    summary = get_harmony_summary()
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write('# HELP music_notes_total Total notes logged\n')
            f.write('# TYPE music_notes_total counter\n')
            f.write(f'music_notes_total {summary.get("total_notes",0)}\n')
            f.write('# HELP music_unique_notes Unique notes\n')
            f.write('# TYPE music_unique_notes gauge\n')
            f.write(f'music_unique_notes {summary.get("unique_notes",0)}\n')
            f.write('# HELP music_harmony_score Harmony score\n')
            f.write('# TYPE music_harmony_score gauge\n')
            f.write(f'music_harmony_score {summary.get("harmony_score",0)}\n')
            f.write('# HELP music_avg_consonance Average consonance\n')
            f.write('# TYPE music_avg_consonance gauge\n')
            f.write(f'music_avg_consonance {summary.get("avg_consonance",0)}\n')
    except Exception:
        pass
    return path

def get_harmony_summary():
    if not SONG_LOG.exists():
        return {}
    entries = []
    with open(SONG_LOG, "r", encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            entries.append(e)
    if not entries:
        return {}
    notes = [e.get("note", 60) for e in entries]
    # Simple harmony stats
    from collections import Counter
    cnt = Counter(notes)
    # Consonance / voice leading
    consonance_sum = 0.0
    voice_leading_sum = 0.0
    for i in range(1, len(notes)):
        interval = abs(notes[i] - notes[i-1])
        consonance_sum += interval_consonance(interval)
        voice_leading_sum += abs(notes[i] - notes[i-1])
    avg_consonance = consonance_sum / max(1, len(notes)-1)
    avg_voice_leading = voice_leading_sum / max(1, len(notes)-1)
    return {
        "total_notes": len(notes),
        "unique_notes": len(cnt),
        "most_common": cnt.most_common(5),
        "avg_consonance": round(avg_consonance,4),
        "avg_voice_leading": round(avg_voice_leading,4),
        "harmony_score": round(avg_consonance * (1 - avg_voice_leading/24.0),4)
    }
