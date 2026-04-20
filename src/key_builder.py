"""Dynamic key builder — core of the peer-aware grading algorithm."""

import numpy as np

from src.vectorizer import ASAGVectorizer
from src.preprocessing import question_word_indices

DEMOTION_FACTOR = 1.8
DEFAULT_STRICTNESS = 20


def build_dynamic_key(
    reference_tokens: list[str],
    corpus_token_lists: list[list[str]],
    vectorizer: ASAGVectorizer,
    question_tokens: list[str] | None = None,
    strictness: int = DEFAULT_STRICTNESS,
) -> np.ndarray:
    """
    Construct the dynamic answer key for one question.

    Algorithm (matches the validated R implementation):
      1. Start with corpus TF / strictness (peer signal).
      2. Demote question-word dimensions by DEMOTION_FACTOR.
      3. Override positions corresponding to reference-answer terms to 1.
         (Reference words take priority over the scaled corpus TF.)

    The override in step 3 — rather than an additive blend — means that
    reference terms always contribute exactly 1.0 to the key regardless of
    corpus frequency, while non-reference terms retain their scaled peer signal.

    Parameters
    ----------
    reference_tokens   : preprocessed (stemmed) tokens of the model answer
    corpus_token_lists : preprocessed tokens for every student response
    vectorizer         : fitted ASAGVectorizer (local, per-question)
    question_tokens    : raw (un-stemmed) question tokens for demotion
    strictness         : peer-signal scaling factor (default 20)

    Returns
    -------
    key : np.ndarray of shape (vocab_size,)
    """
    # Step 1 — scaled corpus TF (peer signal)
    key = vectorizer.corpus_tf(corpus_token_lists) / strictness

    # Step 2 — question-word demotion (applied before reference override)
    if question_tokens:
        indices = question_word_indices(question_tokens, vectorizer.vocabulary_)
        key[indices] /= DEMOTION_FACTOR

    # Step 3 — reference-answer override (set to 1, not add)
    for token in reference_tokens:
        if token in vectorizer.vocabulary_:
            key[vectorizer.vocabulary_[token]] = 1.0

    return key
