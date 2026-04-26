"""
Hot-swappable embedding backends.

Usage:
    from src.embeddings import get_backend

    backend = get_backend("null")               # no-op, F8 excluded
    backend = get_backend("all-MiniLM-L6-v2")  # HuggingFace model on CPU
    backend = get_backend("all-mpnet-base-v2")  # swap to a different model

The active backend is selected at server startup via the EMBED_MODEL env var.
Training and evaluation scripts accept --embed <model_name>.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class EmbeddingBackend(ABC):
    @abstractmethod
    def encode(self, texts: list[str]) -> np.ndarray:
        """Return (n, dim) float32 array of embeddings."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    def is_null(self) -> bool:
        return False


class NullBackend(EmbeddingBackend):
    """No-op backend. When active, F8 is excluded from the feature matrix."""

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.empty((len(texts), 0), dtype=np.float32)

    @property
    def name(self) -> str:
        return "null"

    @property
    def is_null(self) -> bool:
        return True


class SentenceTransformerBackend(EmbeddingBackend):
    """Wraps a HuggingFace sentence-transformers model, forced to CPU."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name, device="cpu")
        self._name = model_name

    def encode(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=64,
        )

    @property
    def name(self) -> str:
        return self._name


def get_backend(name: str = "null") -> EmbeddingBackend:
    """
    Factory. Pass any sentence-transformers model name or "null".

    Examples:
        get_backend("null")
        get_backend("all-MiniLM-L6-v2")
        get_backend("all-mpnet-base-v2")
        get_backend("paraphrase-multilingual-MiniLM-L12-v2")
    """
    if not name or name.strip().lower() == "null":
        return NullBackend()
    return SentenceTransformerBackend(name)
