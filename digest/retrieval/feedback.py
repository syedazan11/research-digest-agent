"""Feedback learning: score papers by how much they look like the ones you liked (Rocchio).

    profile = β · mean(👍 vectors) − γ · mean(👎 vectors)        (TF-IDF vectors)
    score(p) = cosine(p, profile)

Classic Rocchio weights (β = 0.75, γ = 0.25): positives say more about what you want than
negatives do, because "not relevant" covers everything else.

Training examples come from two sources, kept distinguishable by `source`:
- "reaction": your 👍/👎 on digest comments (digest.db feedback table)
- "label":    your y/n hand labels (data/labels.db)
If a paper has both, the reaction wins: it's your judgement of the paper as delivered.

No leakage: when a paper that is itself a training example gets scored, its own vector is removed
from the profile first. A paper's own label can never help it, in production or in the eval.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from digest.models import Paper
from digest.retrieval.tfidf import TfidfVectorizer

BETA = 0.75
GAMMA = 0.25


@dataclass(frozen=True)
class Example:
    paper: Paper
    rating: int  # +1 relevant / 👍, -1 not relevant / 👎
    source: str  # "label" or "reaction"


def merge_examples(reactions: Sequence[Example], labels: Sequence[Example]) -> list[Example]:
    """One example per paper; a reaction overrides a hand label for the same paper."""
    by_id = {e.paper.id: e for e in labels}
    by_id.update({e.paper.id: e for e in reactions})
    return list(by_id.values())


class FeedbackRetriever:
    name = "feedback"

    def __init__(self, examples: Sequence[Example], beta: float = BETA, gamma: float = GAMMA):
        self.examples = list(merge_examples([e for e in examples if e.source == "reaction"],
                                            [e for e in examples if e.source != "reaction"]))
        self.beta = beta
        self.gamma = gamma

    def counts(self) -> dict[str, int]:
        out = {"positive": 0, "negative": 0, "label": 0, "reaction": 0}
        for e in self.examples:
            out["positive" if e.rating > 0 else "negative"] += 1
            out[e.source] = out.get(e.source, 0) + 1
        return out

    def score(self, query: str, papers: Sequence[Paper]) -> list[float]:
        if not papers:
            return []
        if not self.examples:
            return [0.0] * len(papers)  # cold start: no opinion, so no effect in the hybrid

        # One vector space for training examples and the papers being scored. Fitting idf on the
        # target's text is unsupervised (no labels involved), so it isn't leakage.
        texts: dict[str, str] = {e.paper.id: e.paper.text for e in self.examples}
        texts.update({p.id: p.text for p in papers})
        ids = list(texts)
        vec = TfidfVectorizer().fit([texts[i] for i in ids])
        matrix = vec.transform([texts[i] for i in ids])
        row = {pid: i for i, pid in enumerate(ids)}

        pos = [e for e in self.examples if e.rating > 0]
        neg = [e for e in self.examples if e.rating < 0]
        pos_sum = matrix[[row[e.paper.id] for e in pos]].sum(axis=0) if pos else np.zeros(matrix.shape[1])
        neg_sum = matrix[[row[e.paper.id] for e in neg]].sum(axis=0) if neg else np.zeros(matrix.shape[1])
        rating_of = {e.paper.id: e.rating for e in self.examples}

        scores = []
        for p in papers:
            v = matrix[row[p.id]]
            ps, n_pos, ns, n_neg = pos_sum, len(pos), neg_sum, len(neg)
            own = rating_of.get(p.id)
            if own is not None and own > 0:  # leave this paper's own label out
                ps, n_pos = ps - v, n_pos - 1
            elif own is not None:
                ns, n_neg = ns - v, n_neg - 1
            profile = np.zeros_like(v)
            if n_pos:
                profile = profile + self.beta * ps / n_pos
            if n_neg:
                profile = profile - self.gamma * ns / n_neg
            norm = float(np.linalg.norm(profile))
            scores.append(float(v @ profile) / norm if norm > 1e-12 else 0.0)
        return scores
