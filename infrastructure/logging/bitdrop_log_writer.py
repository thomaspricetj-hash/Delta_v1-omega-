"""
BitDrop-v2 canonical log writer / reader
Production-grade append-only compressed logs with .bd2 format
"""
import os
import struct
from pathlib import Path
import zlib
import json

MAGIC = b'BD2\x01'
FRAME_HEADER_SIZE = 4  # uint32 little endian compressed length

class BitDropLogWriter:
    def __init__(self, base_dir: Path, use_rust=True):
        self.base_dir = Path(base_dir)
        self.use_rust = use_rust
        self._bd2 = None
        try:
            import bitdrop_v2 as _bd2
            self._bd2 = _bd2
            self.rust_available = True
        except Exception:
            self.rust_available = False

    def _bd2_path(self, log_path: Path) -> Path:
        p = Path(log_path)
        if p.suffix == '.bd2':
            return p
        # map jsonl -> jsonl.bd2, keep name
        return p.with_suffix(p.suffix + '.bd2')

    def _compress(self, data: bytes) -> bytes:
        if self.rust_available and self.use_rust:
            try:
                # Prefer adaptive
                return self._bd2.compress_adaptive_py(data)
            except Exception:
                pass
        # Fallback to zlib with BitDrop header for compatibility
        return zlib.compress(data)

    def _decompress(self, data: bytes) -> bytes:
        if self.rust_available and self.use_rust:
            try:
                return self._bd2.decompress_py(data)
            except Exception:
                pass
        return zlib.decompress(data)

    def write_line(self, log_path, line: str):
        path = self._bd2_path(Path(log_path))
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = line.encode('utf-8')
        comp = self._compress(raw)
        # Frame: [4-byte comp_len][compressed]
        with open(path, 'ab') as f:
            f.write(struct.pack('<I', len(comp)))
            f.write(comp)

    def append_lines(self, log_path, lines):
        path = self._bd2_path(Path(log_path))
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'ab') as f:
            for line in lines:
                raw = line.encode('utf-8')
                comp = self._compress(raw)
                f.write(struct.pack('<I', len(comp)))
                f.write(comp)

    def read_lines(self, log_path):
        path = self._bd2_path(Path(log_path))
        if not path.exists():
            # fallback to original jsonl
            orig = Path(log_path)
            if orig.exists():
                with open(orig, 'r', encoding='utf-8') as f:
                    for line in f:
                        yield line.rstrip('\n')
                return
            return
        with open(path, 'rb') as f:
            while True:
                hdr = f.read(4)
                if not hdr or len(hdr) < 4:
                    break
                comp_len = struct.unpack('<I', hdr)[0]
                comp = f.read(comp_len)
                if len(comp) < comp_len:
                    break
                try:
                    raw = self._decompress(comp)
                    yield raw.decode('utf-8')
                except Exception:
                    # If decompress fails, skip frame
                    continue

    def exists(self, log_path):
        return self._bd2_path(Path(log_path)).exists()

    @staticmethod
    def migrate_file(src_path: Path, dst_path: Path = None):
        """
        Migrate an uncompressed jsonl file to .bd2 format.
        Returns destination path.
        """
        src = Path(src_path)
        if not src.exists():
            return None
        if dst_path is None:
            dst = src.with_suffix(src.suffix + '.bd2')
        else:
            dst = Path(dst_path)
        writer = BitDropLogWriter(src.parent)
        # Read source lines
        lines = []
        try:
            with open(src, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception:
            return None
        # Write compressed
        writer.append_lines(dst, [l.rstrip('\n') for l in lines])
        return dst

# Global singleton for convenience
_log_writer_instance = None

def get_writer(base_dir=None):
    global _log_writer_instance
    if _log_writer_instance is None:
        if base_dir is None:
            base_dir = Path(__file__).resolve().parents[3] / 'runtime' / 'logs'
        _log_writer_instance = BitDropLogWriter(base_dir)
    return _log_writer_instance
