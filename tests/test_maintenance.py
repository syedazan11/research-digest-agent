from digest.db import Database
from digest.maintenance import commit_message, last_run_id, main, vacuum


def test_last_run_id(tmp_path):
    path = tmp_path / "d.db"
    assert last_run_id(path) == 0  # no DB yet
    with Database(path, default_topic="test-topic") as db:
        db.log_run(1, 1, 1, ["bm25"])
        db.log_run(1, 1, 1, ["bm25"])
    assert last_run_id(path) == 2


def test_commit_message(tmp_path, papers):
    path = tmp_path / "d.db"
    with Database(path, default_topic="test-topic") as db:
        db.save_papers(papers)
        before = db.log_run(10, 10, 10, ["bm25"])
        run = db.log_run(10, 10, 10, ["bm25"])
        db.save_rankings(run, [{"paper_id": p.id, "hybrid_rank": i, "hybrid_score": 0.5,
                                "final_rank": i if i <= 5 else None} for i, p in enumerate(papers, 1)])
        db.save_digest_issue(run, 7, "url")
        db.upsert_feedback(papers[0].id, 1, 123, "2026-11-25T00:00:00Z")

    assert commit_message(path, since_run=before) == f"digest: run {run} (5 papers, issue #7)"
    assert commit_message(path, since_run=run) == "digest: feedback only (1 ratings)"


def test_commit_message_without_issue(tmp_path):
    path = tmp_path / "d.db"
    with Database(path, default_topic="test-topic") as db:
        db.log_run(0, 0, 0, ["bm25"])
    assert commit_message(path, since_run=0) == "digest: run 1 (0 papers)"


def test_vacuum_shrinks_after_deletes(tmp_path):
    path = tmp_path / "d.db"
    with Database(path, default_topic="test-topic") as db:
        db.conn.execute("CREATE TABLE junk (x BLOB)")
        db.conn.executemany("INSERT INTO junk VALUES (?)", [(b"x" * 10_000,)] * 50)
        db.conn.commit()
        db.conn.execute("DELETE FROM junk")
        db.conn.commit()
    before, after = vacuum(path)
    assert after < before


def test_cli(tmp_path, capsys):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("topic:\n  name: T\n  keywords: [k]\ndatabase:\n  path: d.db\n", encoding="utf-8")
    with Database(tmp_path / "d.db", default_topic="test-topic") as db:
        db.log_run(0, 0, 0, ["bm25"])
    assert main(["--config", str(cfg), "last-run"]) == 0
    assert main(["--config", str(cfg), "commit-message", "--since", "0"]) == 0
    assert main(["--config", str(cfg), "vacuum"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "1" and out[1] == "digest: run 1 (0 papers)" and out[2].startswith("VACUUM d.db")
