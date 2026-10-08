"""BM25 retrieval with identifier-aware tokenisation and weighted RRF."""
import re
from functools import lru_cache
from typing import Any

import numpy as np
import snowballstemmer
from rank_bm25 import BM25Okapi

from src.models import MinimalSource

# --- Tunable constants (adjust them with the evaluate command) ---
RRF_K = 60
POOL_SIZE = 50
CODE_QUERY_WEIGHTS: tuple[float, float] = (1.0, 0.95)  # (code, docs)
DOC_QUERY_WEIGHTS: tuple[float, float] = (0.95, 1.0)
BM25_K1 = 1.5
BM25_B = 0.75

STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does",
    "for", "from", "how", "i", "if", "in", "is", "it", "its", "of", "on",
    "or", "that", "the", "this", "to", "what", "when", "where", "which",
    "who", "why", "with", "would", "should", "there", "their",
})
CODE_STOPWORDS = frozenset({
    "self", "def", "return", "import", "none", "true", "false", "class",
    "function", "method",
})
CODE_HINTS = frozenset({"function", "method", "class", "attribute"})
IDENT_RE = re.compile(
    r"\w+_\w+|\b[a-z]+[A-Z][a-z]\w*|\b[A-Z][a-z]+[A-Z][a-z]\w*"
    r"|\w+\(\)|\.py\b"
)

_STEMMER = snowballstemmer.stemmer("english")


@lru_cache(maxsize=None)
def stem(word: str) -> str:
    """Return the English stem of a lowercase word (cached).

    Args:
        word: Lowercase word.

    Returns:
        Its Snowball stem.
    """
    return str(_STEMMER.stemWord(word))


def split_identifier(word: str) -> list[str]:
    """Split snake_case and CamelCase identifiers into sub-words.

    Args:
        word: An identifier such as ``max_num_seqs`` or ``LLMEngine``.

    Returns:
        The list of sub-words (original case preserved).
    """
    subwords: list[str] = []
    for part in word.split("_"):
        if not part:
            continue
        camel_parts = re.findall(
            r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+",
            part)
        subwords.extend(camel_parts if camel_parts else [part])
    return subwords


def tokenize(text: str, is_code: bool = False) -> list[str]:
    """Tokenize text: lowercase, drop stopwords, stem, split identifiers.

    Whole identifiers are kept unstemmed so that verbatim identifier
    queries still match exactly.

    Args:
        text: Text to tokenize.
        is_code: Whether identifier splitting and code stopwords apply.

    Returns:
        The list of tokens.
    """
    raw_tokens = re.sub(r"[^\w\s]", " ", text).split()
    tokens: list[str] = []
    for word in raw_tokens:
        is_ident = is_code and (
            "_" in word or any(c.isupper() for c in word[1:])
        )
        if is_ident:
            tokens.append(word.lower())
            pieces = [w.lower() for w in split_identifier(word)]
        else:
            pieces = [word.lower()]
        for piece in pieces:
            if piece in STOPWORDS:
                continue
            if is_code and piece in CODE_STOPWORDS:
                continue
            tokens.append(stem(piece))
    return tokens


def looks_like_code(query: str) -> bool:
    """Guess whether a question targets source code.

    Args:
        query: The question.

    Returns:
        True if it contains an identifier or a code-related keyword.
    """
    if IDENT_RE.search(query):
        return True
    words = set(re.findall(r"[a-z]+", query.lower()))
    return bool(words & CODE_HINTS)


class Retriever:
    """Two BM25 indices (code / docs) fused with weighted RRF."""

    def __init__(self, indexed_chunks: list[dict[str, Any]]) -> None:
        """Build the BM25 indices from the indexed chunks.

        Args:
            indexed_chunks: Chunks produced by the indexing step.
        """
        self.indexed_chunks = indexed_chunks

        self.code_indices = [
            i for i, c in enumerate(indexed_chunks)
            if c["file_path"].endswith(".py")
        ]
        self.doc_indices = [
            i for i, c in enumerate(indexed_chunks)
            if not c["file_path"].endswith(".py")
        ]

        code_corpus = [
            list(dict.fromkeys(indexed_chunks[i]["tokens"]))
            for i in self.code_indices
        ]
        doc_corpus = [
            indexed_chunks[i]["tokens"] for i in self.doc_indices
        ]

        self.bm25_code = (
            BM25Okapi(code_corpus, k1=BM25_K1, b=BM25_B)
            if code_corpus else None
        )
        self.bm25_docs = (
            BM25Okapi(doc_corpus, k1=BM25_K1, b=BM25_B)
            if doc_corpus else None
        )

    @staticmethod
    def _rank(
            bm25: BM25Okapi | None,
            indices: list[int],
            tokens: list[str]) -> list[int]:
        """Return the chunk ids of the best matches, best first."""
        if bm25 is None or not tokens:
            return []
        scores = bm25.get_scores(tokens)
        top = np.argsort(scores)[::-1][:POOL_SIZE]
        return [indices[int(j)] for j in top if scores[int(j)] > 0]

    def search(self, query: str, k: int) -> list[MinimalSource]:
        """Return the top-k sources for a query.

        Args:
            query: The question.
            k: Number of sources wanted.

        Returns:
            Up to ``k`` sources, best first (empty on degenerate input).
        """
        if k <= 0 or not query.strip():
            return []

        code_rank = self._rank(
            self.bm25_code, self.code_indices,
            tokenize(query, is_code=True))
        doc_rank = self._rank(
            self.bm25_docs, self.doc_indices, tokenize(query))

        if looks_like_code(query):
            w_code, w_doc = CODE_QUERY_WEIGHTS
        else:
            w_code, w_doc = DOC_QUERY_WEIGHTS

        fused: dict[int, float] = {}
        for ranking, weight in ((code_rank, w_code), (doc_rank, w_doc)):
            for rank, idx in enumerate(ranking):
                fused[idx] = fused.get(idx, 0.0) + weight / (
                    RRF_K + rank + 1)

        best = sorted(fused, key=lambda i: fused[i], reverse=True)[:k]
        return [
            MinimalSource(
                file_path=self.indexed_chunks[i]["file_path"],
                first_character_index=(
                    self.indexed_chunks[i]["first_character_index"]),
                last_character_index=(
                    self.indexed_chunks[i]["last_character_index"]),
            )
            for i in best
        ]
