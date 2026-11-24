"""Keyword relevance with BM25: fast, free, offline, explainable."""

from __future__ import annotations

from typing import Sequence

from rank_bm25 import BM25Okapi

from digest.models import Paper
from digest.text import tokenize


class BM25Retriever:
    name = "bm25"

    def score(self, query: str, papers: Sequence[Paper]) -> list[float]:
        if not papers:
            return []
        # BM25Okapi divides by average doc length, so give empty docs a placeholder token
        corpus = [tokenize(p.text) or ["_empty_"] for p in papers]
        bm25 = BM25Okapi(corpus)
        return [float(s) for s in bm25.get_scores(tokenize(query))]
