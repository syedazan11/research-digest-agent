import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import QUERY, RELEVANT_IDS
from digest.retrieval import BM25Retriever, HybridRanker
from digest.retrieval.gemini_embed import GeminiEmbeddingRetriever
from digest.retrieval.hybrid import min_max


# --- BM25 ---
def test_bm25_puts_relevant_papers_on_top(papers):
    scores = BM25Retriever().score(QUERY, papers)
    top4 = {p.id for p, _ in sorted(zip(papers, scores), key=lambda x: -x[1])[:4]}
    assert len(top4 & RELEVANT_IDS) >= 3


def test_bm25_empty():
    assert BM25Retriever().score("anything", []) == []


# --- min-max ---
def test_min_max():
    assert min_max([2, 4, 6]) == [0.0, 0.5, 1.0]
    assert min_max([5, 5]) == [0.0, 0.0]
    assert min_max([]) == []


# --- hybrid ---
class Stub:
    def __init__(self, name, scores):
        self.name, self._scores = name, scores

    def score(self, query, papers):
        return self._scores[: len(papers)]


def test_hybrid_weights_and_top_k(papers):
    three = papers[:3]
    a = Stub("a", [1.0, 0.0, 0.5])  # prefers paper 0
    b = Stub("b", [0.0, 1.0, 0.5])  # prefers paper 1
    ranked = HybridRanker([(a, 0.8), (b, 0.2)]).rank("q", three, top_k=2)
    assert [r.paper.id for r in ranked] == [three[0].id, three[2].id]
    assert ranked[0].score == pytest.approx(0.8)
    assert ranked[0].components == {"a": 1.0, "b": 0.0}


def test_hybrid_rejects_wrong_length(papers):
    with pytest.raises(ValueError):
        HybridRanker([(Stub("bad", [1.0]), 1.0)]).rank("q", papers[:3], top_k=3)


def test_hybrid_needs_a_weighted_retriever():
    with pytest.raises(ValueError):
        HybridRanker([(Stub("a", []), 0.0)])


# --- Gemini embeddings with a fake client (no network) ---
class FakeModels:
    def __init__(self):
        self.calls = 0

    def embed_content(self, model, contents, config):
        self.calls += 1

        def vec(text):  # deterministic fake vector per text
            seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
            return np.random.default_rng(seed).normal(size=config.output_dimensionality).tolist()

        return SimpleNamespace(embeddings=[SimpleNamespace(values=vec(t)) for t in contents])


def test_gemini_retriever_uses_cache(db, papers):
    fake = SimpleNamespace(models=FakeModels())
    r = GeminiEmbeddingRetriever(db, client=fake, dimensions=16, batch_size=4)

    first = r.score(QUERY, papers)
    calls_after_first = fake.models.calls
    assert calls_after_first == 1 + 3  # 1 query + ceil(10 papers / 4 per batch)
    assert all(-1.0 <= s <= 1.0 for s in first)

    second = r.score(QUERY, papers)
    assert fake.models.calls == calls_after_first  # everything came from the cache
    assert second == pytest.approx(first)


# --- a failing retriever is skipped, not fatal ---
class Boom:
    name = "embedding"

    def score(self, query, papers):
        raise RuntimeError("401 UNAUTHENTICATED")


def test_hybrid_skips_failing_retriever(papers, caplog):
    three = papers[:3]
    ok = Stub("bm25", [0.2, 1.0, 0.6])
    ranker = HybridRanker([(ok, 0.4), (Boom(), 0.6)])
    with caplog.at_level("WARNING"):
        ranked = ranker.rank("q", three, top_k=3)

    assert [r.paper.id for r in ranked] == [three[1].id, three[2].id, three[0].id]
    assert ranked[0].score == pytest.approx(1.0)  # weights renormalized over the survivors
    assert ranked[0].components == {"bm25": 1.0}
    assert ranker.active == ["bm25"] and ranker.names == ["bm25", "embedding"]
    assert "Retriever 'embedding' failed" in caplog.text


def test_hybrid_all_retrievers_failing_raises(papers):
    with pytest.raises(RuntimeError, match="every retriever failed"):
        HybridRanker([(Boom(), 1.0)]).rank("q", papers[:3], top_k=3)
