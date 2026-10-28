"""Dependency-free retrieval helpers for consistent BM25 and FAISS fusion."""

import re

# English-book tokenizer, used identically for document and query terms.
TOKEN_RE = re.compile(r"[a-z0-9]+(?:['’-][a-z0-9]+)*", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def reciprocal_rank_fusion(
    dense_ids: list[int], sparse_ids: list[int], k: int = 60
) -> list[int]:
    """Merge two ranked lists using stable chunk IDs, not list positions."""
    scores: dict[int, float] = {}
    for ranking in (dense_ids, sparse_ids):
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))
