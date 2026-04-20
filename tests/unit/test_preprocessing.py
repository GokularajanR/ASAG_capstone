"""UT-PP-01..06: preprocessing module tests."""

import pytest
from src.preprocessing import tokenize, stem_tokens, preprocess, question_word_indices


# UT-PP-01
def test_tokenize_lowercase():
    tokens = tokenize("Hello World")
    assert all(t == t.lower() for t in tokens)


# UT-PP-02
def test_tokenize_removes_punctuation():
    tokens = tokenize("Hello, world! It's great.")
    for t in tokens:
        assert t.isalpha(), f"Non-alpha token found: {t!r}"


# UT-PP-03
def test_tokenize_removes_stopwords():
    tokens = tokenize("the cat sat on the mat")
    stopwords_present = {"the", "on"}
    assert not stopwords_present.intersection(set(tokens))


# UT-PP-04
def test_stem_tokens_applies_stemming():
    stemmed = stem_tokens(["running", "jumps", "easily"])
    assert "run" in stemmed
    assert "jump" in stemmed


# UT-PP-05
def test_preprocess_returns_stemmed_tokens():
    result = preprocess("The dogs are running quickly")
    assert isinstance(result, list)
    assert len(result) > 0
    assert "run" in result or "quickli" in result  # Porter stems


# UT-PP-06
def test_question_word_indices_returns_valid_indices():
    vocab = {"photosynthesi": 0, "plant": 1, "light": 2, "role": 3}
    q_tokens = tokenize("What is the role of photosynthesis?")
    indices = question_word_indices(q_tokens, vocab)
    assert isinstance(indices, list)
    # 'role' and 'photosynthesi' (stemmed) should map to indices 3, 0
    assert set(indices).issubset({0, 1, 2, 3})


def test_preprocess_empty_string():
    assert preprocess("") == []


def test_preprocess_only_stopwords():
    result = preprocess("the and is a")
    assert result == []
