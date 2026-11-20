"""
Production-grade PID + LQR controller for AI Memory system
Controls compression aggressiveness, heart consolidation decay, and reasoning blend weights
based on memory pressure and recall quality metrics.
"""
from __future__ import annotations
import threading
import numpy as np
from dataclasses import dataclass
from typing import Tuple

@dataclass
class PID:
    kp: float
    ki: float
    kd: float
    setpoint: float
    integral: float = 0.0
    prev_err: float = 0.0
    out_min: float = -1.0
    out_max: float = 1.0

    def step(self, measurement: float, dt: float = 1.0) -> float:
        err = self.setpoint - measurement
        self.integral += err * dt
        deriv = (err - self.prev_err) / dt if dt > 0 else 0.0
        self.prev_err = err
        out = self.kp * err + self.ki * self.integral + self.kd * deriv
        return float(max(self.out_min, min(self.out_max, out)))

class LQR:
    """Simple discrete LQR for 2-state system: [size, recall] -> [compress, decay_adj]"""
    def __init__(self, Q: np.ndarray, R: np.ndarray, A: np.ndarray, B: np.ndarray):
        self.Q = Q
        self.R = R
        self.A = A
        self.B = B
        self.K = self._solve_dare()
    
    def _solve_dare(self, max_iter=100, tol=1e-6):
        P = self.Q.copy()
        for _ in range(max_iter):
            P_next = self.Q + self.A.T @ P @ self.A - self.A.T @ P @ self.B @ np.linalg.inv(self.R + self.B.T @ P @ self.B) @ self.B.T @ P @ self.A
            if np.allclose(P, P_next, atol=tol):
                break
            P = P_next
        K = np.linalg.inv(self.R + self.B.T @ P @ self.B) @ self.B.T @ P @ self.A
        return K

    def control(self, state: np.ndarray) -> np.ndarray:
        return -self.K @ state

class MemoryController:
    """
    PID+LQR controller for AI Memory orchestrator
    Controls:
    - compression aggressiveness
    - heart consolidation decay
    - reasoning blend weights
    """
    def __init__(self):
        self.lock = threading.RLock()
        # PID for average delta chain length, target 6-8
        self.compress_pid = PID(kp=0.2, ki=0.01, kd=0.05, setpoint=7.0, out_min=0.0, out_max=1.0)
        # PID for recall quality, target 0.75
        self.recall_pid = PID(kp=0.15, ki=0.005, kd=0.03, setpoint=0.75, out_min=-0.2, out_max=0.2)
        # LQR for joint size/recall tradeoff
        # A: state transition approx, B: control effect
        A = np.array([[0.95, 0.0], [0.0, 0.98]])
        B = np.array([[0.3, 0.0], [-0.1, 0.2]])
        Q = np.diag([1.0, 2.0])  # penalize size and recall drop
        R = np.diag([0.1, 0.1])
        self.lqr = LQR(Q, R, A, B)

        # Reasonable bounds
        self.decay_min = 0.985
        self.decay_max = 0.999

    def compute(self, avg_chain_len: float, recall_score: float, memory_size_norm: float) -> Tuple[float, float, float]:
        """
        Returns:
        compress_aggressiveness in [0,1]
        decay_adjustment delta
        recall_weight_adjustment
        """
        with self.lock:
            # PID outputs
            compress_u = self.compress_pid.step(avg_chain_len)
            recall_u = self.recall_pid.step(recall_score)

            # LQR joint control
            state = np.array([memory_size_norm, 1.0 - recall_score])
            lqr_u = self.lqr.control(state)
            # lqr_u[0] -> compress, lqr_u[1] -> decay adjust
            compress_lqr = float(np.clip(0.5 + 0.5 * lqr_u[0], 0.0, 1.0))
            decay_adj_lqr = float(np.clip(lqr_u[1], -0.01, 0.01))

            # Blend PID and LQR
            compress = 0.6 * compress_u + 0.4 * compress_lqr
            decay_adj = 0.7 * recall_u * 0.01 + 0.3 * decay_adj_lqr

            return compress, decay_adj, recall_u

    def recommend_decay(self, base_decay: float, decay_adj: float) -> float:
        new_decay = base_decay + decay_adj
        return float(max(self.decay_min, min(self.decay_max, new_decay)))

    def recommend_compress_threshold(self, base_threshold: int, compress_aggr: float) -> int:
        # More aggressive -> lower threshold
        thresh = int(base_threshold * (1.0 - 0.5 * compress_aggr))
        return max(4, thresh)
