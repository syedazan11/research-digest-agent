"""TF-IDF and the Rocchio feedback retriever, including the leakage guard."""

from dataclasses import replace

import numpy as np
import pytest

from conftest import QUERY, RELEVANT_IDS
from digest.retrieval import BM25Retriever, HybridRanker
from digest.retrieval.feedback import Example, FeedbackRetriever, merge_examples
from digest.retrieval.tfidf import TfidfRetriever, TfidfVectorizer


# --- TF-IDF ---
def test_tfidf_rows_are_unit_vectors_and_identical_texts_match():
    vec = TfidfVectorizer().fit(["judge models evaluate", "protein folding", "judge models evaluate"])
    m = vec.transform(["judge models evaluate", "protein folding", "judge models evaluate", "zzz unknown"])
    assert np.allclose(np.linalg.norm(m[:3], axis=1), 1.0)
    assert m[0] @ m[2] == pytest.approx(1.0) and m[0] @ m[1] == pytest.approx(0.0)
    assert not m[3].any()  # unseen words only: zero vector, not NaN


def test_tfidf_retriever_puts_relevant_papers_on_top(papers):
    scores = TfidfRetriever().score(QUERY, papers)
    top4 = {p.id for p, _ in sorted(zip(papers, scores), key=lambda x: -x[1])[:4]}
    assert len(top4 & RELEVANT_IDS) >= 3
    assert TfidfRetriever().score(QUERY, []) == []


# --- feedback / Rocchio ---
def ex(paper, rating, source="label"):
    return Example(paper, rating, source)


def test_cold_start_scores_zero(papers):
    assert FeedbackRetriever([]).score("q", papers) == [0.0] * len(papers)


def test_liked_papers_pull_similar_ones_up(papers):
    by_id = {p.id: p for p in papers}
    # like two RAG-evaluation papers, dislike protein folding
    fb = FeedbackRetriever([ex(by_id["sample-001"], 1), ex(by_id["sample-003"], 1), ex(by_id["sample-005"], -1)])
    unseen = [by_id["sample-002"], by_id["sample-004"], by_id["sample-006"], by_id["sample-010"]]
    scores = dict(zip((p.id for p in unseen), fb.score("q", unseen)))
    assert scores["sample-002"] > scores["sample-006"] and scores["sample-002"] > scores["sample-010"]


def test_own_label_never_helps_the_paper(papers):
    """Leakage guard: the only 👍 is paper X itself. Scoring X must not see its own label."""
    x = papers[0]
    twin = replace(x, id="twin-of-x")  # same text, different paper
    fb = FeedbackRetriever([ex(x, 1)])
    own, twin_score = fb.score("q", [x, twin])
    assert own == 0.0  # profile without X is empty: no opinion
    assert twin_score == pytest.approx(1.0)  # the same label does help *other* papers


def test_own_negative_label_never_hurts_the_paper(papers):
    x, y = papers[0], papers[1]
    fb = FeedbackRetriever([ex(x, -1), ex(y, 1)])
    alone = FeedbackRetriever([ex(y, 1)]).score("q", [x])[0]
    assert fb.score("q", [x])[0] == pytest.approx(alone)  # x's own 👎 is removed before scoring x


def test_leave_one_out_matches_retraining_without_the_paper(papers):
    examples = [ex(p, 1 if p.id in RELEVANT_IDS else -1) for p in papers]
    full = FeedbackRetriever(examples).score("q", papers)
    for i, p in enumerate(papers):
        without = FeedbackRetriever([e for e in examples if e.paper.id != p.id])
        assert full[i] == pytest.approx(without.score("q", papers)[i], abs=1e-9)


def test_reaction_overrides_label_and_counts(papers):
    x, y = papers[0], papers[1]
    merged = merge_examples([ex(x, -1, "reaction")], [ex(x, 1, "label"), ex(y, 1, "label")])
    assert {(e.paper.id, e.rating, e.source) for e in merged} == {(x.id, -1, "reaction"), (y.id, 1, "label")}
    fb = FeedbackRetriever([ex(x, 1, "label"), ex(x, -1, "reaction"), ex(y, 1, "label")])
    assert fb.counts() == {"positive": 1, "negative": 1, "label": 1, "reaction": 1}


def test_feedback_plugs_into_the_hybrid(papers):
    by_id = {p.id: p for p in papers}
    fb = FeedbackRetriever([ex(by_id["sample-002"], 1), ex(by_id["sample-004"], 1)])
    ranked = HybridRanker([(BM25Retriever(), 0.6), (fb, 0.4)]).rank(QUERY, papers, top_k=10)
    assert {"bm25", "feedback"} == set(ranked[0].components)


# --- production training data ---
def test_load_examples_merges_reactions_and_labels_read_only(tmp_path, db, papers):
    from digest.labels import LabelStore
    from digest.retrieval.feedback import load_examples

    a, b, c = papers[:3]
    db.save_papers([a, b])
    db.upsert_feedback(a.id, -1, 111, "2026-11-25T00:00:00Z")  # 👎 on a
    path = tmp_path / "labels.db"
    with LabelStore(path, default_topic="test-topic") as s:
        s.add_to_pool([a, c])
        s.set_label(a.id, True)  # label says relevant, but the reaction wins
        s.set_label(c.id, True)
    before = path.read_bytes()

    got = {(e.paper.id, e.rating, e.source) for e in load_examples(db, path)}
    assert got == {(a.id, -1, "reaction"), (c.id, 1, "label")}
    assert path.read_bytes() == before  # opened read-only
    assert {e.source for e in load_examples(db, tmp_path / "missing.db")} == {"reaction"}
