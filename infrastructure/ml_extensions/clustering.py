"""
Production-grade clustering for motifs and frames.
KMeans and DBSCAN with robust initialization.
"""
from __future__ import annotations
from typing import Tuple
import numpy as np

try:
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False
    KMeans = None
    DBSCAN = None
    StandardScaler = None


class MotifClusterer:
    def __init__(self, n_clusters: int = 4096, eps: float = 0.5, min_samples: int = 5, random_state: int = 42):
        self.n_clusters = n_clusters
        self.eps = eps
        self.min_samples = min_samples
        self.random_state = random_state
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None

    def kmeans(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn not available")
        Xs = self.scaler.fit_transform(X) if self.scaler else X
        km = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        labels = km.fit_predict(Xs)
        return labels, km.cluster_centers_

    def dbscan(self, X: np.ndarray) -> np.ndarray:
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn not available")
        Xs = self.scaler.fit_transform(X) if self.scaler else X
        db = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        return db.fit_predict(Xs)
