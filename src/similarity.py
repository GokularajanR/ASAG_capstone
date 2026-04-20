"""Similarity between a student response and the dynamic key."""

import numpy as np

from src.vectorizer import ASAGVectorizer


def compute_similarity(
    response_tokens: list[str],
    key_vec: np.ndarray,
    vectorizer: ASAGVectorizer,
) -> float:
    """
    Compute similarity between a student response and the dynamic key.

    Response vector = binary presence vector + TF-IDF weights.

    Normalization matches the validated R implementation:
        similarity = dot(l1, l2) / sqrt(sum(l1) * sum(l2))
    where sum() is the L1 norm (sum of elements, not squared).
    This differs from standard cosine similarity which uses L2 norms.

    The result is multiplied by 5 to place it on the 0-5 score scale,
    matching the R `score_vect = score_vect * 5` step.

    Returns 0.0 for empty or zero-sum vectors.
    """
    binary_vec = vectorizer.binary_vector(response_tokens)
    tfidf_vec = vectorizer.transform([response_tokens])[0]

    response_vec = binary_vec + tfidf_vec

    sum_r = float(np.sum(response_vec))
    sum_k = float(np.sum(key_vec))

    if sum_r == 0.0 or sum_k == 0.0:
        return 0.0

    dot = float(np.dot(response_vec, key_vec))
    sim = dot / np.sqrt(sum_r * sum_k)
    return float(np.clip(sim * 5.0, 0.0, 5.0))


def batch_similarity(
    token_lists: list[list[str]],
    key_vec: np.ndarray,
    vectorizer: ASAGVectorizer,
) -> list[float]:
    """Compute similarity for a list of responses against one key."""
    return [compute_similarity(toks, key_vec, vectorizer) for toks in token_lists]
