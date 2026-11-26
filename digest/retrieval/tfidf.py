"""TF-IDF vectors and a TF-IDF retriever: the classic baseline, and the vector space feedback learning uses.

Hand-rolled in numpy (~40 lines) instead of scikit-learn: same maths, no heavy dependency on CI.
Sublinear tf (1 + log count) and smoothed idf (log((1+n)/(1+df)) + 1), rows L2-normalized,
so a dot product between two rows is their cosine similarity.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Sequence

import numpy as np

from digest.models import Paper
from digest.text import tokenize


class TfidfVectorizer:
    def __init__(self):
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray = np.zeros(0)

    def fit(self, texts: Sequence[str]) -> "TfidfVectorizer":
        docs = [set(tokenize(t)) for t in texts]
        df = Counter(tok for d in docs for tok in d)
        self.vocab = {tok: i for i, tok in enumerate(sorted(df))}
        n = len(texts)
        self.idf = np.array([math.log((1 + n) / (1 + df[t])) + 1.0 for t in sorted(df)], dtype=np.float64)
        return self

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        m = np.zeros((len(texts), len(self.vocab)), dtype=np.float64)
        for row, text in enumerate(texts):
            for tok, c in Counter(tokenize(text)).items():
                j = self.vocab.get(tok)
                if j is not None:
                    m[row, j] = (1.0 + math.log(c)) * self.idf[j]
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        return np.divide(m, norms, out=np.zeros_like(m), where=norms > 0)


class TfidfRetriever:
    name = "tfidf"

    def score(self, query: str, papers: Sequence[Paper]) -> list[float]:
        if not papers:
            return []
        vec = TfidfVectorizer().fit([p.text for p in papers])
        docs = vec.transform([p.text for p in papers])
        q = vec.transform([query])[0]
        return [float(s) for s in docs @ q]
