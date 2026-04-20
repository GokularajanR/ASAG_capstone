"""
Multi-feature extraction for one question group.

Returns a (n_responses, 6) matrix:
  F1  peer-aware sim WITH question demotion  (0-5 scale)
  F2  peer-aware sim WITHOUT question demotion (0-5 scale)
  F3  length ratio  response_tokens / reference_tokens
  F4  Jaccard overlap of stemmed unigrams
  F5  keyword coverage  |response ∩ reference| / |reference|
  F6  bigram Jaccard on raw lowercased text
"""

import re
import numpy as np

from src.key_builder import build_dynamic_key
from src.preprocessing import preprocess, tokenize
from src.similarity import compute_similarity
from src.vectorizer import ASAGVectorizer

_PUNCT = re.compile(r"[^\w\s]")


def _bigrams(text: str) -> set[str]:
    words = _PUNCT.sub(" ", text.lower()).split()
    return {f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)}


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def extract_features(
    question: str,
    responses: list[str],
    reference: str,
    strictness: int = 20,
) -> np.ndarray:
    """
    Compute the 6-feature matrix for all responses to one question.

    Returns ndarray of shape (len(responses), 6).
    """
    question_tokens  = tokenize(question)
    reference_tokens = preprocess(reference)
    corpus_tokens    = [preprocess(r) for r in responses]
    ref_set          = set(reference_tokens)

    vectorizer = ASAGVectorizer()
    vectorizer.fit(corpus_tokens)

    if vectorizer.vocab_size == 0:
        return np.zeros((len(responses), 6))

    # Keys
    key_with    = build_dynamic_key(reference_tokens, corpus_tokens, vectorizer, question_tokens, strictness)
    key_without = build_dynamic_key(reference_tokens, corpus_tokens, vectorizer, None,            strictness)

    ref_bigrams = _bigrams(reference)
    ref_len     = max(len(reference_tokens), 1)

    rows = []
    for tok, raw in zip(corpus_tokens, responses):
        f1 = compute_similarity(tok, key_with,    vectorizer)
        f2 = compute_similarity(tok, key_without, vectorizer)
        f3 = len(tok) / ref_len
        f4 = _jaccard(set(tok), ref_set)
        f5 = len(set(tok) & ref_set) / ref_len
        f6 = _jaccard(_bigrams(raw), ref_bigrams)
        rows.append([f1, f2, f3, f4, f5, f6])

    return np.array(rows, dtype=float)


FEATURE_NAMES = [
    "sim_with_demotion",
    "sim_no_demotion",
    "length_ratio",
    "jaccard_unigram",
    "keyword_coverage",
    "jaccard_bigram",
]
