"""Text preprocessing: normalisation, stemming, question-word demotion."""

import re
import string
from functools import lru_cache

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Ensure required NLTK data is present
for _pkg in ("stopwords", "punkt_tab", "punkt"):
    try:
        nltk.data.find(f"corpora/{_pkg}" if _pkg == "stopwords" else f"tokenizers/{_pkg}")
    except LookupError:
        nltk.download(_pkg, quiet=True)

_STOP_WORDS = set(stopwords.words("english"))
_STEMMER = PorterStemmer()
_PUNCT_RE = re.compile(r"[^\w\s]")


def _clean(text: str) -> str:
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    return text


def tokenize(text: str) -> list[str]:
    """Lowercase, remove punctuation, tokenize, remove stopwords, keep alpha tokens."""
    cleaned = _clean(text)
    tokens = word_tokenize(cleaned)
    return [t for t in tokens if t.isalpha() and t not in _STOP_WORDS]


def stem_tokens(tokens: list[str]) -> list[str]:
    """Apply Porter stemming to a token list."""
    return [_STEMMER.stem(t) for t in tokens]


def preprocess(text: str) -> list[str]:
    """Full pipeline: tokenize → remove stopwords → stem."""
    return stem_tokens(tokenize(text))


def question_word_indices(
    question_tokens: list[str],
    vocabulary: dict[str, int],
) -> list[int]:
    """
    Return vocabulary indices of stemmed question tokens.
    Used to identify positions in the key vector to demote.
    """
    stemmed_q = stem_tokens(question_tokens)
    return [vocabulary[t] for t in stemmed_q if t in vocabulary]
