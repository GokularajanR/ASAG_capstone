"""
Multi-feature extraction for one question group.

Feature matrix shape: (n_responses, N) where N depends on flags:
  F1  peer-aware sim WITH question demotion  (0-5 scale)
  F2  peer-aware sim WITHOUT question demotion (0-5 scale)
  F3  length ratio  response_tokens / reference_tokens
  F4  Jaccard overlap of stemmed unigrams
  F5  keyword coverage  |response ∩ reference| / |reference|
  F6  bigram Jaccard on raw lowercased text
  F7  has_any_overlap — 1.0 if F4>0 or F5>0, else 0.0  (use_f7=True)
  F8  embedding_angle — arccos(cosine) between response and reference
                        embeddings in radians, range [0, π]  (embedding_backend != null)
"""

from __future__ import annotations

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


def _angular_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Angle in radians between two vectors: arccos(cosine). Range [0, π].
    Returns π for zero vectors (maximally distant)."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0.0:
        return float(np.pi)
    cos_sim = float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))
    return float(np.arccos(cos_sim))


def extract_features(
    question: str,
    responses: list[str],
    reference: str,
    strictness: int = 20,
    use_f7: bool = True,
    embedding_backend=None,
) -> np.ndarray:
    """
    Compute the feature matrix for all responses to one question.

    Parameters
    ----------
    use_f7             : include F7 (has_any_overlap binary feature)
    embedding_backend  : EmbeddingBackend instance; F8 included when not NullBackend.
                         Pass None or NullBackend() to exclude F8.
    """
    use_f8 = embedding_backend is not None and not embedding_backend.is_null

    question_tokens  = tokenize(question)
    reference_tokens = preprocess(reference)
    corpus_tokens    = [preprocess(r) for r in responses]
    ref_set          = set(reference_tokens)

    vectorizer = ASAGVectorizer()
    vectorizer.fit(corpus_tokens)

    n_features = 6 + int(use_f7) + int(use_f8)
    if vectorizer.vocab_size == 0:
        return np.zeros((len(responses), n_features))

    key_with    = build_dynamic_key(reference_tokens, corpus_tokens, vectorizer, question_tokens, strictness)
    key_without = build_dynamic_key(reference_tokens, corpus_tokens, vectorizer, None, strictness)

    ref_bigrams = _bigrams(reference)
    ref_len     = max(len(reference_tokens), 1)

    # Compute embeddings once for the whole batch (F8)
    angles: list[float] = []
    if use_f8:
        all_texts  = responses + [reference]
        embeddings = embedding_backend.encode(all_texts)
        ref_emb    = embeddings[-1]
        angles     = [_angular_distance(embeddings[i], ref_emb) for i in range(len(responses))]

    rows = []
    for idx, (tok, raw) in enumerate(zip(corpus_tokens, responses)):
        f1 = compute_similarity(tok, key_with,    vectorizer)
        f2 = compute_similarity(tok, key_without, vectorizer)
        f3 = len(tok) / ref_len
        f4 = _jaccard(set(tok), ref_set)
        f5 = len(set(tok) & ref_set) / ref_len
        f6 = _jaccard(_bigrams(raw), ref_bigrams)
        row = [f1, f2, f3, f4, f5, f6]
        if use_f7:
            row.append(1.0 if (f4 > 0 or f5 > 0) else 0.0)
        if use_f8:
            row.append(angles[idx])
        rows.append(row)

    return np.array(rows, dtype=float)


FEATURE_NAMES = [
    "sim_with_demotion",
    "sim_no_demotion",
    "length_ratio",
    "jaccard_unigram",
    "keyword_coverage",
    "jaccard_bigram",
    "has_any_overlap",   # F7 — only when use_f7=True
    "embedding_angle",   # F8 — only when embedding backend is not null
]
