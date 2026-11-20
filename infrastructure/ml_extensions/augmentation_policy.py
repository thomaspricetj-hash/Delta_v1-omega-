"""
Production-grade Q-Learning policy optimizer for augmentation scheduling.
Controls rotation_prob and speed_factor in train_weave_decoder.
"""
from __future__ import annotations
import numpy as np
from collections import defaultdict
from typing import Tuple

class QLearningPolicy:
    def __init__(self, n_states: int = 8, n_actions: int = 3, alpha: float = 0.1, gamma: float = 0.95, epsilon: float = 0.2):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = defaultdict(lambda: np.zeros(n_actions))
        self.state_bins = np.linspace(0.0, 2.0, n_states + 1)

    def _discretize(self, loss_ratio: float) -> int:
        # loss_ratio = ema_loss / loss_target
        clipped = np.clip(loss_ratio, 0.0, 2.0)
        idx = int(np.digitize([clipped], self.state_bins)[0]) - 1
        return max(0, min(self.n_states - 1, idx))

    def select_action(self, loss_ratio: float) -> int:
        state = self._discretize(loss_ratio)
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.q_table[state]))

    def update(self, loss_ratio: float, action: int, reward: float, next_loss_ratio: float):
        state = self._discretize(loss_ratio)
        next_state = self._discretize(next_loss_ratio)
        best_next = np.max(self.q_table[next_state])
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error

    def action_to_params(self, action: int) -> Tuple[float, float]:
        # Map action to rotation_prob and speed_factor
        # 0: conservative, 1: balanced, 2: aggressive
        params = [
            (0.3, 1.0),
            (0.5, 1.2),
            (0.7, 1.5),
        ]
        return params[action]


class AugmentationPolicy:
    def __init__(self):
        self.q = QLearningPolicy()

    def step(self, ema_loss: float, loss_target: float, prev_loss: float | None) -> Tuple[float, float]:
        loss_ratio = ema_loss / (loss_target + 1e-8)
        # Reward: negative loss change, penalize high variance
        reward = 0.0
        if prev_loss is not None:
            delta = prev_loss - ema_loss
            reward = delta  # positive if loss decreased
        action = self.q.select_action(loss_ratio)
        # Simulate next state with dummy, in real loop update after step
        rot_prob, speed = self.q.action_to_params(action)
        return rot_prob, speed, action, reward

    def update(self, ema_loss: float, loss_target: float, prev_loss: float | None, action: int, reward: float, next_ema_loss: float):
        loss_ratio = ema_loss / (loss_target + 1e-8)
        next_loss_ratio = next_ema_loss / (loss_target + 1e-8)
        self.q.update(loss_ratio, action, reward, next_loss_ratio)
