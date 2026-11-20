"""
LQR Reward Controller for Matrix Multiply Predictor
Full production-grade integration:
- Discrete Algebraic Riccati Equation solver
- Per-op state: [confidence_error, reward_error, penalty]
- Control: [reward_increment, penalty_decrement]
- Automatic gain scheduling from EMA
- Prometheus export
- Anomaly logging
"""
import json
import time
from pathlib import Path
import numpy as np

BASE = Path(__file__).resolve().parents[2]
PROM_PATH = BASE / "runtime" / "logs" / "lqr_reward.prom"
ANOMALY_PATH = BASE / "runtime" / "logs" / "lqr_anomalies.jsonl"

# System matrices - discrete time
# x_{t+1} = A x_t + B u_t
A = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0]
], dtype=float)

B = np.array([
    [0.001, 0.0],
    [0.01, 0.0],
    [0.0, 0.02]
], dtype=float)

# Cost matrices - tuned for smooth convergence
Q_base = np.diag([10.0, 1.0, 0.1])
R_base = np.diag([0.01, 0.01])

def solve_discrete_are(A, B, Q, R, max_iter=1000, tol=1e-9):
    P = Q.copy()
    for _ in range(max_iter):
        P_next = Q + A.T @ P @ A - A.T @ P @ B @ np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
        if np.max(np.abs(P_next - P)) < tol:
            break
        P = P_next
    K = np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
    return K, P

class LQRRewardController:
    def __init__(self):
        self.ops = ['+','-','*','/']
        self.K = {}
        self.P = {}
        self._init_gains()
        self._ema_alpha = 0.1
        self._state = {op: np.zeros(3) for op in self.ops}

    def _init_gains(self):
        for op in self.ops:
            # Gain scheduling by op difficulty
            if op == '/':
                Q = Q_base * 1.2
            elif op == '*':
                Q = Q_base * 1.1
            else:
                Q = Q_base
            K, P = solve_discrete_are(A, B, Q, R_base)
            self.K[op] = K
            self.P[op] = P

    def update(self, op, stats, ema=None):
        # State definition with EMA gain scheduling
        conf_target = 0.99
        reward_target = 1.5
        # EMA-based target adjustment: higher EMA -> tighter target
        ema_factor = 1.0
        if ema is not None and op in ema:
            ema_factor = 0.5 + 0.5 * ema[op]  # 0.5 to 1.0
            conf_target = 0.95 + 0.04 * ema[op]
        conf_err = conf_target - stats.get('confidence', 0.85)
        rew_err = reward_target - stats.get('reward', 1.0)
        penalty = stats.get('penalty', 0.0)
        x = np.array([conf_err, rew_err, penalty], dtype=float)

        # EMA smoothing of state
        prev = self._state[op]
        x_smooth = self._ema_alpha * x + (1 - self._ema_alpha) * prev
        self._state[op] = x_smooth

        K = self.K[op]
        u = -K @ x_smooth
        # Clamp control to safe ranges
        reward_inc = float(np.clip(u[0], 0, 0.1))
        penalty_dec = float(np.clip(u[1], 0, 0.05))

        # Apply with small step size for stability
        new_reward = min(1.5, stats.get('reward',1.0) + 0.01*reward_inc)
        new_penalty = max(0.0, stats.get('penalty',0.0) - 0.005*penalty_dec)
        new_conf = min(0.99, stats.get('confidence',0.85) + 0.001*reward_inc)

        # Anomaly detection
        if abs(conf_err) > 0.2 or penalty > 0.8:
            try:
                with open(ANOMALY_PATH,'a',encoding='utf-8') as f:
                    f.write(json.dumps({
                        'ts': time.time(),
                        'op': op,
                        'conf_err': float(conf_err),
                        'rew_err': float(rew_err),
                        'penalty': float(penalty),
                        'u': u.tolist()
                    }) + '\n')
            except Exception:
                pass

        # Prometheus export
        self._export(op, new_reward, new_penalty, new_conf, u)

        return {
            'reward': new_reward,
            'penalty': new_penalty,
            'confidence': new_conf,
            'u': u.tolist()
        }

    def _export(self, op, reward, penalty, conf, u):
        try:
            with open(PROM_PATH,'w',encoding='utf-8') as f:
                f.write('# HELP lqr_reward_op Reward value\n')
                f.write('# TYPE lqr_reward_op gauge\n')
                f.write(f'lqr_reward_op{{op="{op}"}} {reward}\n')
                f.write('# HELP lqr_penalty_op Penalty value\n')
                f.write('# TYPE lqr_penalty_op gauge\n')
                f.write(f'lqr_penalty_op{{op="{op}"}} {penalty}\n')
                f.write('# HELP lqr_confidence_op Confidence value\n')
                f.write('# TYPE lqr_confidence_op gauge\n')
                f.write(f'lqr_confidence_op{{op="{op}"}} {conf}\n')
                f.write('# HELP lqr_control_u Control input\n')
                f.write('# TYPE lqr_control_u gauge\n')
                f.write(f'lqr_control_u{{op="{op}",idx="0"}} {u[0]}\n')
                f.write(f'lqr_control_u{{op="{op}",idx="1"}} {u[1]}\n')
        except Exception:
            pass

# Singleton
_controller = LQRRewardController()

def update_reward_lqr(op, stats, ema=None):
    return _controller.update(op, stats, ema)
