"""UT-SIM-01..04: similarity module tests."""

import numpy as np
import pytest
from src.vectorizer import ASAGVectorizer
from src.key_builder import build_dynamic_key
from src.similarity import compute_similarity, batch_similarity

CORPUS = [
    ["photosynthesi", "plant", "light", "glucos"],
    ["plant", "convert", "sunlight", "energi"],
    ["chlorophyl", "absorb", "light"],
]
REFERENCE = ["photosynthesi", "plant", "light", "glucos"]
QUESTION_TOKENS = ["what", "photosynthesi"]


@pytest.fixture
def setup():
    v = ASAGVectorizer()
    v.fit(CORPUS + [REFERENCE])
    key = build_dynamic_key(REFERENCE, CORPUS, v, QUESTION_TOKENS)
    return v, key


# UT-SIM-01
def test_similarity_range(setup):
    v, key = setup
    sim = compute_similarity(CORPUS[0], key, v)
    assert 0.0 <= sim <= 5.0  # feature is scaled x5 to match 0-5 score range


# UT-SIM-02
def test_identical_response_high_similarity(setup):
    v, key = setup
    sim = compute_similarity(REFERENCE, key, v)
    assert sim > 1.0, f"Expected high similarity for reference, got {sim:.4f}"


# UT-SIM-03
def test_empty_response_returns_zero(setup):
    v, key = setup
    sim = compute_similarity([], key, v)
    assert sim == 0.0


# UT-SIM-04
def test_zero_key_returns_zero(setup):
    v, key = setup
    zero_key = np.zeros_like(key)
    sim = compute_similarity(CORPUS[0], zero_key, v)
    assert sim == 0.0


def test_batch_similarity_length(setup):
    v, key = setup
    sims = batch_similarity(CORPUS, key, v)
    assert len(sims) == len(CORPUS)
    assert all(0.0 <= s <= 5.0 for s in sims)


def test_relevant_response_higher_than_irrelevant(setup):
    v, key = setup
    sim_relevant  = compute_similarity(["photosynthesi", "plant", "light"], key, v)
    sim_irrelevant = compute_similarity(["car", "road", "drive"], key, v)
    # irrelevant tokens not in vocab → similarity should be ≤ relevant
    assert sim_relevant >= sim_irrelevant
