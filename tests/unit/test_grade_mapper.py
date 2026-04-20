"""Unit tests for GradeMapper (GBM, 6-feature input)."""

import numpy as np
import pytest
from src.grade_mapper import GradeMapper

N_FEATURES = 6


def _make_X(n=40):
    rng = np.random.default_rng(0)
    return rng.uniform(0, 1, (n, N_FEATURES))


def _make_scores(X):
    # Simple monotone target: first feature drives grade
    return np.clip(X[:, 0] * 5, 0, 5)


@pytest.fixture
def fitted_mapper():
    X = _make_X(80)
    y = _make_scores(X)
    return GradeMapper().fit(X, y)


def test_predict_in_range(fitted_mapper):
    X = _make_X(20)
    for row in X:
        p = fitted_mapper.predict(row)
        assert 0.0 <= p <= 5.0


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        GradeMapper().predict(np.zeros(N_FEATURES))


def test_predict_batch_length(fitted_mapper):
    X = _make_X(10)
    preds = fitted_mapper.predict_batch(X)
    assert len(preds) == 10


def test_predict_batch_in_range(fitted_mapper):
    X = _make_X(20)
    preds = fitted_mapper.predict_batch(X)
    assert all(0.0 <= p <= 5.0 for p in preds)


def test_save_and_load(tmp_path, fitted_mapper):
    path = tmp_path / "mapper.joblib"
    fitted_mapper.save(path)
    loaded = GradeMapper.load(path)
    X = _make_X(5)
    for row in X:
        assert abs(loaded.predict(row) - fitted_mapper.predict(row)) < 1e-6


def test_feature_importances(fitted_mapper):
    imp = fitted_mapper.feature_importances_
    assert imp.shape == (N_FEATURES,)
    assert abs(imp.sum() - 1.0) < 1e-6
