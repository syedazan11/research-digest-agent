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
