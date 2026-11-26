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


def test_in_memory_copy_reads_disk_but_never_writes(tmp_path, papers):
    from digest.db import Database

    path = tmp_path / "real.db"
    with Database(path, default_topic="test-topic") as real:
        real.save_papers(papers[:3])
        real.put_embedding("p1", "m", 3, np.array([1, 2, 3], dtype=np.float32))
    disk = path.read_bytes()

    with Database(path, in_memory_copy=True, default_topic="test-topic") as dry:
        assert dry.count_papers() == 3  # sees what's on disk (dedup still works)
        assert dry.get_embedding("p1", "m", 3) is not None  # cache hits still work
        dry.save_papers(papers)
        dry.log_run(1, 1, 1, ["bm25"])
        assert dry.count_papers() == len(papers)
    assert path.read_bytes() == disk  # file untouched

    missing = tmp_path / "sub" / "none.db"
    with Database(missing, in_memory_copy=True, default_topic="test-topic") as dry:
        assert dry.count_papers() == 0
    assert not missing.parent.exists()  # not even the folder is created
