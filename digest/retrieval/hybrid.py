"""Combine several retrievers into one ranking.

Each retriever's raw scores are min-max normalized to [0, 1] (BM25 scores might run 0-15,
cosine similarity 0.4-0.8), then mixed with the weights from config.yaml.

If one retriever throws (API down, bad key), it is skipped with a warning and the weights are
renormalized over the ones that worked, so one outage degrades the ranking instead of killing the run.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

from digest.models import Paper
from digest.retrieval.base import Retriever

log = logging.getLogger(__name__)


def min_max(scores: Sequence[float]) -> list[float]:
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-12:
        return [0.0] * len(scores)  # all equal = this retriever can't tell papers apart
    return [(s - lo) / (hi - lo) for s in scores]


@dataclass
class RankedPaper:
    paper: Paper
    score: float  # final hybrid score in [0, 1]
    components: dict[str, float] = field(default_factory=dict)  # normalized score per retriever

    def to_dict(self, rank: int) -> dict:
        return {
            "rank": rank,
            "score": round(self.score, 4),
            "components": {k: round(v, 4) for k, v in self.components.items()},
            "paper": self.paper.to_dict(),
        }


class HybridRanker:
    def __init__(self, weighted: list[tuple[Retriever, float]]):
        weighted = [(r, w) for r, w in weighted if w > 0]
        if not weighted:
            raise ValueError("HybridRanker needs at least one retriever with weight > 0")
        self.weighted = weighted
        self.active: list[str] = self.names  # retrievers that actually ran in the last rank() call

    @property
    def names(self) -> list[str]:
        return [r.name for r, _ in self.weighted]

    def rank(self, query: str, papers: Sequence[Paper], top_k: int) -> list[RankedPaper]:
        if not papers:
            return []
        normalized: dict[str, list[float]] = {}
        used: list[tuple[Retriever, float]] = []
        for retriever, w in self.weighted:
            try:
                raw = retriever.score(query, papers)
            except Exception as e:  # outage or bad key: skip this retriever, keep the others
                log.warning("Retriever '%s' failed (%s: %s): continuing without it",
                            retriever.name, type(e).__name__, str(e)[:200])
                continue
            if len(raw) != len(papers):  # a code bug, not an outage: fail loudly
                raise ValueError(f"{retriever.name} returned {len(raw)} scores for {len(papers)} papers")
            normalized[retriever.name] = min_max(raw)
            used.append((retriever, w))

        self.active = [r.name for r, _ in used]
        if not used:
            raise RuntimeError("every retriever failed, nothing to rank with")
        total_w = sum(w for _, w in used)

        ranked = []
        for i, p in enumerate(papers):
            comps = {name: scores[i] for name, scores in normalized.items()}
            final = sum(w * comps[r.name] for r, w in used) / total_w
            ranked.append(RankedPaper(paper=p, score=final, components=comps))

        # best score first; ties go to the newer paper
        ranked.sort(key=lambda rp: (rp.score, rp.paper.published), reverse=True)
        return ranked[:top_k]
