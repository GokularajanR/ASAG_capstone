"""GBM grade mapper: (n_features,) → predicted grade [0, max_score]."""

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

DEFAULT_MAX_SCORE = 5.0
DEFAULT_MODEL_PATH = Path("grade_mapper.joblib")

GBM_PARAMS = dict(
    n_estimators=100,
    max_depth=2,
    learning_rate=0.1,
    subsample=0.8,
    min_samples_leaf=10,
    random_state=42,
)


class GradeMapper:
    """
    Gradient Boosting regressor mapping a multi-feature vector to a grade.

    Input : (n_samples, n_features) — see src/features.py for feature layout.
    Output: clipped to [0, max_score].
    """

    def __init__(self, max_score: float = DEFAULT_MAX_SCORE, **gbm_kwargs) -> None:
        self.max_score = max_score
        params = {**GBM_PARAMS, **gbm_kwargs}
        self._model = GradientBoostingRegressor(**params)
        self._fitted = False

    def fit(self, X: np.ndarray, scores: list[float] | np.ndarray) -> "GradeMapper":
        X = np.asarray(X, dtype=float)
        y = np.asarray(scores, dtype=float)
        self._model.fit(X, y)
        self._fitted = True
        return self

    def predict(self, x: np.ndarray) -> float:
        if not self._fitted:
            raise RuntimeError("GradeMapper must be fitted before calling predict().")
        x = np.asarray(x, dtype=float).reshape(1, -1)
        return float(np.clip(self._model.predict(x)[0], 0.0, self.max_score))

    def predict_batch(self, X: np.ndarray) -> list[float]:
        if not self._fitted:
            raise RuntimeError("GradeMapper must be fitted before calling predict_batch().")
        X = np.asarray(X, dtype=float)
        return np.clip(self._model.predict(X), 0.0, self.max_score).tolist()

    @property
    def feature_importances_(self) -> np.ndarray:
        return self._model.feature_importances_

    def save(self, path: str | Path = DEFAULT_MODEL_PATH) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path = DEFAULT_MODEL_PATH) -> "GradeMapper":
        return joblib.load(path)
