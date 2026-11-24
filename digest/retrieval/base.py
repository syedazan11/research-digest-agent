"""The interface every retriever follows.

Adding a new method (TF-IDF, MiniLM, ONNX...) later = one new class with these two members.
Nothing else in the pipeline has to change. That's what makes the comparison experiment cheap.
"""

from __future__ import annotations

from typing import Protocol, Sequence

from digest.models import Paper


class Retriever(Protocol):
    name: str  # short id used in score breakdowns and the runs table, e.g. "bm25"

    def score(self, query: str, papers: Sequence[Paper]) -> list[float]:
        """Return one relevance score per paper, same order as `papers`. Higher = more relevant.
        Scores don't need to be on any fixed scale; HybridRanker normalizes them."""
        ...
