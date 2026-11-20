"""
SMTile - Core compute unit
Integrates SRAM, OverlayChain, DaxController, MicroFiberScheduler, CollapseUnit, DriftMonitor.
Production grade, deterministic iteration.
"""
from __future__ import annotations
from typing import List
import numpy as np
from .overlay_chain import OverlaySram, OverlayChain
from .collapse_unit import CollapseUnit
from .dax_controller import DaxController, DeltaCommand
from .micro_fiber_scheduler import MicroFiberScheduler
from .drift_monitor import DriftMonitor

class SMTile:
    """
    SM-Tile core compute unit.
    Each iteration: Collapse -> Delta shaping -> Fiber scheduling -> Fast path compute -> Drift monitoring.
    """
    def __init__(
        self,
        sram_size: int = 4096,
        num_warps: int = 32,
        warp_size: int = 4,
        max_overlays: int = 1024,
    ):
        self.sram = OverlaySram(size=sram_size)
        self.chain = OverlayChain(max_overlays=max_overlays)
        self.collapse = CollapseUnit(self.chain)
        self.dax = DaxController(num_warps=num_warps)
        self.scheduler = MicroFiberScheduler(num_warps=num_warps, warp_size=warp_size)
        self.drift = DriftMonitor(window_size=1024, threshold=0.15)
        self.iteration = 0
        self._last_overlay_id = None

    def write_sram(self, data: np.ndarray) -> int:
        return self.sram.write(data)

    def step(self, region_id: str, delta_input: np.ndarray) -> dict:
        """
        Single tile iteration.
        Returns metrics dict.
        """
        # 1. Collapse SRAM -> OverlayChain
        oid = self.collapse.collapse(self.sram, layer=0)
        self._last_overlay_id = oid

        # 2. Delta shaping
        warp_id = self.iteration % self.scheduler.num_warps
        cmd: DeltaCommand = self.dax.shape_delta(region_id, delta_input, warp_id, active=True)

        # 3. Fiber scheduling - submit fast path compute
        def fast_path_compute(rid, payload):
            # SIMD-friendly placeholder: apply delta to SRAM buffer
            # In production, this would be AVX2/AVX512 kernel
            arr = self.sram.read(self.sram.size)
            # ReLU-gated delta update for stability and sparsity
            if len(payload) > 0:
                n = min(len(arr), len(payload))
                # ReLU gating: only positive deltas propagate
                relu_payload = np.maximum(payload[:n], 0)
                arr[:n] += relu_payload * 0.01

        self.scheduler.submit(fast_path_compute, region_id, cmd.delta, priority=int(128 * cmd.hotness + 1))

        # 4. Execute a batch of fibers
        executed = self.scheduler.step(max_fibers=64)

        # 5. Drift monitoring
        current_vec = self.sram.read(self.sram.size)
        drift_score = self.drift.update(current_vec)

        self.iteration += 1
        return {
            "iteration": self.iteration,
            "overlay_id": oid,
            "delta_magnitude": float(cmd.magnitude),
            "hotness": float(cmd.hotness),
            "warp_id": int(cmd.warp_id),
            "fibers_executed": executed,
            "drift_score": float(drift_score),
            "is_drifting": bool(self.drift.is_drifting()),
            "pending_fibers": self.scheduler.pending(),
        }

    def reset(self):
        self.sram.reset()
        self.chain.clear()
        self.drift.reset()
        self.iteration = 0
        self.scheduler.reset_stats()

    def hot_overlays(self, k: int = 8):
        return self.chain.get_hotness(k)
