import numpy as np


def test_filter_new_drops_saved_papers(db, papers):
    assert len(db.filter_new(papers)) == len(papers)
    db.save_papers(papers[:3])
    new = db.filter_new(papers)
    assert len(new) == len(papers) - 3
    assert {p.id for p in papers[:3]}.isdisjoint(p.id for p in new)


def test_save_is_idempotent(db, papers):
    db.save_papers(papers)
    db.save_papers(papers)
    assert db.count_papers() == len(papers)


def test_embedding_roundtrip(db):
    v = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    assert db.get_embedding("p1", "m", 3) is None
    db.put_embedding("p1", "m", 3, v)
    np.testing.assert_allclose(db.get_embedding("p1", "m", 3), v)
    assert db.get_embedding("p1", "m", 768) is None  # different dims = different cache entry


def test_log_run(db):
    assert db.log_run(10, 7, 5, ["bm25"]) == 1
    assert db.log_run(3, 0, 0, ["bm25", "embedding"]) == 2


def test_rankings_roundtrip(db):
    run_id = db.log_run(10, 10, 2, ["bm25"], notes="rerank: fake")
    db.save_rankings(run_id, [
        {"paper_id": "a", "hybrid_rank": 1, "hybrid_score": 0.9, "llm_score": 3.0, "reason": "meh", "final_rank": 2},
        {"paper_id": "b", "hybrid_rank": 2, "hybrid_score": 0.8, "llm_score": 9.0, "reason": "great", "final_rank": 1},
        {"paper_id": "c", "hybrid_rank": 3, "hybrid_score": 0.1, "llm_score": None, "reason": "", "final_rank": None},
    ])
    rows = db.get_rankings(run_id)
    assert [r["paper_id"] for r in rows] == ["a", "b", "c"]
    assert [r["selected"] for r in rows] == [1, 1, 0]
    assert rows[2]["llm_score"] is None and rows[2]["reason"] is None
    assert db.get_rankings(run_id + 1) == []
