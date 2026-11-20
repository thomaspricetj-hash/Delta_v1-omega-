"""
Migrate existing uncompressed logs to BitDrop-v2 compressed .bd2 format
Production grade backfill
"""
import sys
from pathlib import Path
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from infrastructure.logging.bitdrop_log_writer import BitDropLogWriter

LOG_DIR = BASE / 'runtime' / 'logs'

def is_already_compressed(p: Path) -> bool:
    return p.suffix == '.bd2'

def migrate_one(p: Path):
    if is_already_compressed(p):
        return None
    if p.is_dir():
        return None
    # Only migrate text logs
    if p.suffix not in {'.jsonl', '.log', '.txt', '.json'}:
        return None
    # Skip empty
    if p.stat().st_size == 0:
        return None
    writer = BitDropLogWriter(LOG_DIR)
    try:
        # Migrate file in place: create .bd2 and optionally remove original
        dst = writer._bd2_path(p)
        # Avoid overwriting existing
        if dst.exists():
            # Check if already migrated
            return dst
        # Read lines
        lines = []
        # Try binary for log files
        try:
            with open(p, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
        except Exception:
            return None
        writer.append_lines(dst, [l.rstrip('\n') for l in lines])
        # Keep original for now, but mark migrated
        print(f"Migrated {p.relative_to(BASE)} -> {dst.relative_to(BASE)} ({len(lines)} lines)")
        return dst
    except Exception as e:
        print(f"Failed {p}: {e}")
        return None

def main():
    count = 0
    total_lines = 0
    for p in LOG_DIR.rglob('*'):
        if p.is_file() and not is_already_compressed(p):
            dst = migrate_one(p)
            if dst:
                count += 1
    print(f"Migration complete. {count} files migrated.")
    # Verify predictor_promotion_summary.json
    summary_path = LOG_DIR / 'predictor_promotion_summary.json'
    if summary_path.exists():
        import json
        try:
            data = json.loads(summary_path.read_text())
            print(f"predictor_promotion_summary.json contains {len(data)} entries")
        except Exception as e:
            print(f"Could not read summary: {e}")
    else:
        print("predictor_promotion_summary.json not found")

if __name__ == '__main__':
    main()
