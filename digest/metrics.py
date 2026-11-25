"""Ranking metrics with binary relevance.

- Precision@k: of the top k, how many are relevant? (What the digest's top 5 actually delivers.)
- Recall@k:    of all relevant papers, how many made the top k? (Did the top 20 sent to the LLM
               contain the good ones? Reranking can't fix what retrieval missed.)
- NDCG@k:      rewards relevant papers near the top: gain 1/log2(rank + 1), divided by the best
               possible ordering, so 1.0 = every relevant paper ranked above every other one.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence


def precision_at_k(ranking: Sequence[str], relevant: Mapping[str, int], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be >= 1")
    top = ranking[:k]
    return sum(relevant.get(pid, 0) for pid in top) / k


def recall_at_k(ranking: Sequence[str], relevant: Mapping[str, int], k: int) -> float:
    total = sum(1 for v in relevant.values() if v)
    if total == 0:
        return 0.0
    return sum(relevant.get(pid, 0) for pid in ranking[:k]) / total


def dcg_at_k(gains: Sequence[int], k: int) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains[:k]))


def ndcg_at_k(ranking: Sequence[str], relevant: Mapping[str, int], k: int) -> float:
    gains = [relevant.get(pid, 0) for pid in ranking]
    ideal = sorted((v for v in relevant.values()), reverse=True)
    best = dcg_at_k(ideal, k)
    return dcg_at_k(gains, k) / best if best > 0 else 0.0
