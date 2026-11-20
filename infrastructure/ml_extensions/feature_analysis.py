"""
Production-grade feature analysis: PCA and ICA for motif/frame embeddings.
"""
from __future__ import annotations
from typing import Tuple
import numpy as np

try:
    from sklearn.decomposition import PCA, FastICA
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False
    PCA = None
    FastICA = None


class FeatureAnalyzer:
    def __init__(self, n_components: int = 32, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self.pca = PCA(n_components=n_components, random_state=random_state) if SKLEARN_AVAILABLE else None
        self.ica = FastICA(n_components=n_components, random_state=random_state, max_iter=500) if SKLEARN_AVAILABLE else None

    def fit_pca(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not SKLEARN_AVAILABLE or self.pca is None:
            raise RuntimeError("scikit-learn not available")
        X_centered = X - X.mean(axis=0)
        X_pca = self.pca.fit_transform(X_centered)
        return X_pca, self.pca.components_

    def transform_pca(self, X: np.ndarray) -> np.ndarray:
        if not SKLEARN_AVAILABLE or self.pca is None:
            raise RuntimeError("scikit-learn not available")
        return self.pca.transform(X - self.pca.mean_)

    def fit_ica(self, X: np.ndarray) -> np.ndarray:
        if not SKLEARN_AVAILABLE or self.ica is None:
            raise RuntimeError("scikit-learn not available")
        return self.ica.fit_transform(X)

    @staticmethod
    def explained_variance_ratio(pca: PCA) -> np.ndarray:
        return pca.explained_variance_ratio_
