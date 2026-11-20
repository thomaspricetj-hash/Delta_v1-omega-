"""
Production-grade data sanitization for TRANSLATION_OS_V1
Z-score + Isolation Forest outlier detection for corpus and delta-state logs.
"""
from __future__ import annotations
from typing import List, Iterable
import numpy as np

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False
    IsolationForest = None
    StandardScaler = None


class DataSanitizer:
    """
    CPU-friendly outlier detection for text windows and delta logs.
    Uses Z-score for fast filtering and Isolation Forest for robust detection.
    """

    def __init__(self, z_thresh: float = 3.0, iforest_contamination: float = 0.01, random_state: int = 42):
        self.z_thresh = z_thresh
        self.iforest_contamination = iforest_contamination
        self.random_state = random_state
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.iforest = None
        if SKLEARN_AVAILABLE:
            self.iforest = IsolationForest(
                contamination=self.iforest_contamination,
                random_state=self.random_state,
                n_estimators=100,
                max_samples=256,
            )

    def _zscore_mask(self, X: np.ndarray) -> np.ndarray:
        if X.size == 0:
            return np.array([], dtype=bool)
        if X.ndim != 2:
            X = np.atleast_2d(X)
        # Row-wise Z-score across features
        means = X.mean(axis=1, keepdims=True)
        stds = X.std(axis=1, keepdims=True) + 1e-9
        z = np.abs((X - means) / stds)
        return (z > self.z_thresh).any(axis=1)

    def fit(self, X: np.ndarray):
        if not SKLEARN_AVAILABLE:
            return
        if X.size == 0:
            # Nothing to fit
            return
        # Ensure 2D
        if X.ndim != 2:
            X = np.atleast_2d(X)
        if self.scaler is not None:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X
        self.iforest.fit(X_scaled)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Returns boolean mask where True = inlier, False = outlier.
        """
        if X.size == 0:
            return np.array([], dtype=bool)
        # Ensure 2D
        if X.ndim != 2:
            X = np.atleast_2d(X)
        # Z-score fast filter
        z_mask = self._zscore_mask(X)
        if not SKLEARN_AVAILABLE or self.iforest is None:
            return ~z_mask
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        iso_pred = self.iforest.predict(X_scaled) == 1
        return (~z_mask) & iso_pred

    @staticmethod
    def window_features(window_ids: List[int]) -> np.ndarray:
        # Compact feature vector for a token window
        arr = np.asarray(window_ids, dtype=np.float32)
        if arr.size == 0:
            return np.zeros(8)
        feats = [
            float(arr.size),
            float(arr.mean()) if arr.size else 0.0,
            float(arr.std()) if arr.size > 1 else 0.0,
            float(np.unique(arr).size / max(1, arr.size)),
            float(np.max(arr) - np.min(arr)) if arr.size else 0.0,
            float((arr[1:] - arr[:-1]).mean()) if arr.size > 1 else 0.0,
            float((arr[1:] - arr[:-1]).std()) if arr.size > 1 else 0.0,
            float(np.sum(arr > 50000) / max(1, arr.size)),
        ]
        return np.array(feats, dtype=np.float32)

    def sanitize_windows(self, windows: List[List[int]]) -> List[List[int]]:
        if not windows:
            return []
        X = np.vstack([self.window_features(w) for w in windows])
        self.fit(X)
        mask = self.predict(X)
        return [w for w, keep in zip(windows, mask) if keep]
