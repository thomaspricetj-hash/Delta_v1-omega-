"""
Production-grade meta-predictors for training control.
LogisticRegression, RandomForest, SVM, DecisionTree.
"""
from __future__ import annotations
from typing import Tuple
import numpy as np

try:
    from sklearn.linear_model import LogisticRegression, Lasso, LinearRegression
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False
    LogisticRegression = None
    RandomForestClassifier = None
    SVC = None
    DecisionTreeClassifier = None
    Lasso = None
    LinearRegression = None
    RandomForestRegressor = None
    DecisionTreeRegressor = None
    StandardScaler = None


class MetaPredictor:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None

    def fit_classifier(self, X: np.ndarray, y: np.ndarray, model_type: str = "logistic") -> object:
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn not available")
        Xs = self.scaler.fit_transform(X) if self.scaler else X
        if model_type == "logistic":
            model = LogisticRegression(max_iter=1000, random_state=self.random_state)
        elif model_type == "random_forest":
            model = RandomForestClassifier(n_estimators=200, random_state=self.random_state)
        elif model_type == "svm":
            model = SVC(probability=True, random_state=self.random_state)
        elif model_type == "decision_tree":
            model = DecisionTreeClassifier(random_state=self.random_state)
        else:
            raise ValueError(f"Unknown model_type {model_type}")
        model.fit(Xs, y)
        return model

    def fit_regressor(self, X: np.ndarray, y: np.ndarray, model_type: str = "lasso") -> object:
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn not available")
        Xs = self.scaler.fit_transform(X) if self.scaler else X
        if model_type == "lasso":
            model = Lasso(alpha=0.001, random_state=self.random_state, max_iter=5000)
        elif model_type == "linear":
            model = LinearRegression()
        elif model_type == "random_forest":
            model = RandomForestRegressor(n_estimators=200, random_state=self.random_state)
        elif model_type == "decision_tree":
            model = DecisionTreeRegressor(random_state=self.random_state)
        else:
            raise ValueError(f"Unknown model_type {model_type}")
        model.fit(Xs, y)
        return model

    def predict(self, model, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler.transform(X) if self.scaler else X
        return model.predict(Xs)
