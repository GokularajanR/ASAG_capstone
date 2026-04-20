"""UT-VEC-01..04: vectorizer module tests."""

import numpy as np
import pytest
from src.vectorizer import ASAGVectorizer


CORPUS = [
    ["photosynthesi", "plant", "light", "glucos"],
    ["plant", "convert", "sunlight", "energi"],
    ["chlorophyl", "absorb", "light", "carbon"],
]


@pytest.fixture
def fitted_vectorizer():
    v = ASAGVectorizer()
    v.fit(CORPUS)
    return v


# UT-VEC-01
def test_fit_builds_vocabulary(fitted_vectorizer):
    assert len(fitted_vectorizer.vocabulary_) > 0
    assert fitted_vectorizer.vocab_size == len(fitted_vectorizer.vocabulary_)


# UT-VEC-02
def test_transform_shape(fitted_vectorizer):
    matrix = fitted_vectorizer.transform(CORPUS)
    assert matrix.shape == (3, fitted_vectorizer.vocab_size)


# UT-VEC-03
def test_binary_vector_presence(fitted_vectorizer):
    tokens = ["plant", "light"]
    vec = fitted_vectorizer.binary_vector(tokens)
    assert vec.shape == (fitted_vectorizer.vocab_size,)
    for t in tokens:
        if t in fitted_vectorizer.vocabulary_:
            assert vec[fitted_vectorizer.vocabulary_[t]] == 1.0


# UT-VEC-04
def test_binary_vector_unknown_token(fitted_vectorizer):
    vec = fitted_vectorizer.binary_vector(["unknownxyz"])
    assert np.sum(vec) == 0.0


def test_tf_vector_counts(fitted_vectorizer):
    tokens = ["plant", "plant", "light"]
    vec = fitted_vectorizer.tf_vector(tokens)
    plant_idx = fitted_vectorizer.vocabulary_.get("plant")
    if plant_idx is not None:
        assert vec[plant_idx] == 2.0


def test_corpus_tf_aggregates(fitted_vectorizer):
    agg = fitted_vectorizer.corpus_tf(CORPUS)
    assert agg.shape == (fitted_vectorizer.vocab_size,)
    assert np.sum(agg) > 0
