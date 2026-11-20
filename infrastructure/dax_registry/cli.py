"""
DAX Registry stats CLI
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from infrastructure.dax_registry.dax_registry import DAXRegistry

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--path', required=True, help='DAX registry base path')
    p.add_argument('--json', action='store_true')
    args = p.parse_args()
    reg = DAXRegistry(Path(args.path))
    stats = reg.stats()
    # Count files
    files = list(Path(args.path).glob('*.bd'))
    total_disk = sum(f.stat().st_size for f in files)
    out = {
        'path': str(args.path),
        'deltas': stats['deltas'],
        'compressed_bytes_in_memory': stats['compressed_bytes'],
        'disk_bytes': total_disk,
        'files': [f.name for f in files]
    }
    if args.json:
        print(json.dumps(out, indent=2))
    else:
        print(f"DAX Registry at {args.path}")
        print(f"Deltas: {out['deltas']}")
        print(f"Disk bytes: {out['disk_bytes']}")
        print(f"Files: {len(out['files'])}")

if __name__ == '__main__':
    main()
