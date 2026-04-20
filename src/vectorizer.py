"""TF-IDF vectorizer wrapper with binary-vector and raw-TF helpers."""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import issparse


class ASAGVectorizer:
    """
    Wraps scikit-learn TfidfVectorizer and exposes helpers needed by the
    dynamic key builder and similarity calculator.

    The vectorizer is fitted on preprocessed token strings (space-joined stems).
    sublinear_tf=False keeps raw TF so corpus aggregation in the key builder
    is straightforward to interpret.
    """

    def __init__(self) -> None:
        self._tfidf = TfidfVectorizer(
            sublinear_tf=False,
            norm="l2",
            analyzer="word",
            token_pattern=r"(?u)\b\w+\b",
        )
        self.vocabulary_: dict[str, int] = {}
        self.vocab_size: int = 0

    # ------------------------------------------------------------------
    def fit(self, token_lists: list[list[str]]) -> "ASAGVectorizer":
        """Fit on a corpus given as lists of (already-stemmed) tokens."""
        docs = [" ".join(toks) for toks in token_lists]
        self._tfidf.fit(docs)
        self.vocabulary_ = self._tfidf.vocabulary_
        self.vocab_size = len(self.vocabulary_)
        return self

    def transform(self, token_lists: list[list[str]]) -> np.ndarray:
        """Return dense TF-IDF matrix (n_docs × vocab_size)."""
        docs = [" ".join(toks) for toks in token_lists]
        mat = self._tfidf.transform(docs)
        return mat.toarray() if issparse(mat) else mat

    # ------------------------------------------------------------------
    def binary_vector(self, tokens: list[str]) -> np.ndarray:
        """1 where token present in vocabulary, 0 elsewhere."""
        vec = np.zeros(self.vocab_size, dtype=float)
        for t in tokens:
            if t in self.vocabulary_:
                vec[self.vocabulary_[t]] = 1.0
        return vec

    def tf_vector(self, tokens: list[str]) -> np.ndarray:
        """Raw term-frequency vector (unnormalised)."""
        vec = np.zeros(self.vocab_size, dtype=float)
        for t in tokens:
            if t in self.vocabulary_:
                vec[self.vocabulary_[t]] += 1.0
        return vec

    def corpus_tf(self, token_lists: list[list[str]]) -> np.ndarray:
        """Sum of raw TF vectors across all documents in a corpus."""
        agg = np.zeros(self.vocab_size, dtype=float)
        for toks in token_lists:
            agg += self.tf_vector(toks)
        return agg
