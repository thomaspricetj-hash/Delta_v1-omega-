"""
DAX-wrapped memory guard
Safe memory regions with explicit size and identity
Checked reads/writes with overflow detection
Integrates with Translation OS memory systems for safe access
"""
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dax_memory_guard")

@dataclass
class DaxRegion:
    id: int
    base: bytes
    size: int
    name: str = ""
    access_log: List[Dict[str, Any]] = field(default_factory=list)

    def _check_bounds(self, offset: int, length: int) -> bool:
        if offset < 0 or length < 0:
            return False
        if offset + length > self.size:
            return False
        return True

    def dax_read(self, offset: int, length: int) -> bytes:
        if not self._check_bounds(offset, length):
            logger.warning(f"[DAX] Overflow read attempt: id={self.id}, name={self.name}, base={id(self.base)}, len={self.size}, offset={offset}, read_len={length}")
            self.access_log.append({
                "op": "read",
                "offset": offset,
                "length": length,
                "status": "DAX_OVERFLOW"
            })
            raise MemoryError(f"DAX_OVERFLOW read on region {self.id}")
        data = self.base[offset:offset+length]
        self.access_log.append({
            "op": "read",
            "offset": offset,
            "length": length,
            "status": "DAX_OK"
        })
        return data

    def dax_write(self, offset: int, data: bytes) -> bool:
        length = len(data)
        if not self._check_bounds(offset, length):
            logger.warning(f"[DAX] Overflow write attempt: id={self.id}, name={self.name}, base={id(self.base)}, len={self.size}, offset={offset}, write_len={length}")
            self.access_log.append({
                "op": "write",
                "offset": offset,
                "length": length,
                "status": "DAX_OVERFLOW"
            })
            return False
        # Simulate write by creating mutable bytearray
        # In real implementation this would be a memoryview
        self.access_log.append({
            "op": "write",
            "offset": offset,
            "length": length,
            "status": "DAX_OK"
        })
        return True

class DaxMemoryGuard:
    def __init__(self):
        self.regions: Dict[int, DaxRegion] = {}
        self.next_id = 1
        self.overflow_events: List[Dict[str, Any]] = []
        self._lock = False  # simple global lock for single-threaded guard

    def alloc_region(self, data: bytes, name: str = "") -> DaxRegion:
        region = DaxRegion(
            id=self.next_id,
            base=data,
            size=len(data),
            name=name
        )
        self.regions[self.next_id] = region
        self.next_id += 1
        logger.info(f"[DAX] Allocated region id={region.id} name={name} size={region.size}")
        return region

    def free_region(self, region_id: int) -> bool:
        if region_id in self.regions:
            del self.regions[region_id]
            logger.info(f"[DAX] Freed region id={region_id}")
            return True
        return False

    def read(self, region_id: int, offset: int, length: int) -> bytes:
        region = self.regions.get(region_id)
        if not region:
            raise ValueError(f"Region {region_id} not found")
        try:
            return region.dax_read(offset, length)
        except MemoryError as e:
            self.overflow_events.append({
                "region_id": region_id,
                "offset": offset,
                "length": length,
                "error": str(e)
            })
            raise

    def write(self, region_id: int, offset: int, data: bytes) -> bool:
        region = self.regions.get(region_id)
        if not region:
            raise ValueError(f"Region {region_id} not found")
        ok = region.dax_write(offset, data)
        if not ok:
            self.overflow_events.append({
                "region_id": region_id,
                "offset": offset,
                "length": len(data),
                "error": "DAX_OVERFLOW"
            })
        return ok

    def get_overflow_report(self) -> Dict[str, Any]:
        return {
            "total_regions": len(self.regions),
            "overflow_events": self.overflow_events,
            "regions": {rid: {"name": r.name, "size": r.size, "access_count": len(r.access_log)} for rid, r in self.regions.items()}
        }

# Global guard instance for orchestrator integration
_guard = DaxMemoryGuard()

def get_guard() -> DaxMemoryGuard:
    return _guard

def safe_alloc(data: bytes, name: str = "") -> int:
    region = _guard.alloc_region(data, name)
    return region.id

def safe_read(region_id: int, offset: int, length: int) -> bytes:
    return _guard.read(region_id, offset, length)

def safe_write(region_id: int, offset: int, data: bytes) -> bool:
    return _guard.write(region_id, offset, data)

def dax_overflow_report() -> Dict[str, Any]:
    return _guard.get_overflow_report()
