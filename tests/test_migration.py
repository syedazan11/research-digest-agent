"""Schema 1 (single topic) -> 2 (multi-topic): nothing lost, everything moved to the first topic."""

import json
import sqlite3
from pathlib import Path

import pytest

from digest.db import SCHEMA_VERSION, Database, MigrationError

SCHEMA_V1 = (Path(__file__).parent / "fixtures" / "schema_v1.sql").read_text(encoding="utf-8")
SUMMARY = json.dumps({"problem": "p", "method": "m", "key_result": "k", "why_it_matters": "w", "limitations": "l"})


def make_v1(path: Path) -> None:
    """A small schema-1 database shaped like the real one after run 5."""
    c = sqlite3.connect(path)
    c.executescript(SCHEMA_V1)
    for pid, sent in (("2611.1", "2026-11-25"), ("2611.2", None), ("2611.3", "2026-11-25")):
        c.execute("INSERT INTO papers VALUES (?, 'T', 'A', '[]', '2026-11-20T00:00:00+00:00', '[]', '', '', "
                  "'2026-11-24T00:00:00+00:00', ?)", (pid, sent))
    c.execute("INSERT INTO runs (started_at, n_fetched, n_new, n_ranked, retrievers) "
              "VALUES ('2026-11-25T00:00:00+00:00', 3, 3, 3, '[\"bm25\"]')")
    c.execute("INSERT INTO rankings VALUES (1, '2611.1', 1, 0.9, 9, 'r', 1, 1)")
    c.execute("INSERT INTO summaries VALUES ('2611.1', 'groq/m', ?, 'pdf', NULL, '2026-11-25T00:00:00+00:00')",
              (SUMMARY,))
    c.execute("INSERT INTO digest_issues VALUES (1, 3, 'url', '2026-11-25', '2026-11-25', NULL)")
    c.execute("INSERT INTO deliveries VALUES (1, '2611.1', 3, 111, '2026-11-25T00:00:00+00:00')")
    c.execute("INSERT INTO feedback (paper_id, rating, created_at, comment_id, source, updated_at) "
              "VALUES ('2611.1', 1, '2026-11-25T09:00:00Z', 111, 'github', '2026-11-25')")
    c.commit()
    c.close()


def counts(path: Path) -> dict:
    c = sqlite3.connect(path)
    out = {t: c.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
           for t in ("papers", "runs", "rankings", "summaries", "digest_issues", "deliveries", "feedback")}
    c.close()
    return out


def test_migration_keeps_everything_and_moves_it_to_the_first_topic(tmp_path):
    path = tmp_path / "digest.db"
    make_v1(path)
    before = counts(path)

    with Database(path, default_topic="llm-evaluation") as db:
        assert db.conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        assert counts(path) == before  # no row lost
        assert db.run_topic(1) == "llm-evaluation"
        assert db.get_digest_issue(1)["topic"] == "llm-evaluation"
        assert db.get_feedback("2611.1", "llm-evaluation")["rating"] == 1
        assert db.get_summary("2611.1", "groq/m", "llm-evaluation") is not None
        assert db.get_summary("2611.1", "groq/m", "ai-agents") is None  # per topic from now on
        # the old "seen" set became llm-evaluation's, with sent_in_digest carried over
        tp = dict(db.conn.execute("SELECT paper_id, sent_in_digest FROM topic_papers WHERE topic = 'llm-evaluation'"))
        assert tp == {"2611.1": "2026-11-25", "2611.2": None, "2611.3": "2026-11-25"}
        assert db.known_ids(["2611.1", "2611.9"], "llm-evaluation") == {"2611.1"}
        assert db.known_ids(["2611.1"], "ai-agents") == set()  # a new topic has seen nothing
    assert (tmp_path / "digest.db.bak-schema1").exists()  # backup written first


def test_migration_is_idempotent(tmp_path):
    path = tmp_path / "digest.db"
    make_v1(path)
    with Database(path, default_topic="llm-evaluation"):
        pass
    snapshot = counts(path)
    with Database(path, default_topic="something-else") as db:  # already migrated: topic arg not used for data
        assert db.run_topic(1) == "llm-evaluation"
    assert counts(path) == snapshot


def test_old_data_without_a_topic_is_refused_not_guessed(tmp_path):
    path = tmp_path / "digest.db"
    make_v1(path)
    with pytest.raises(MigrationError, match="which topic"):
        Database(path)
    c = sqlite3.connect(path)
    assert "topic" not in {r[1] for r in c.execute("PRAGMA table_info(runs)")}  # left untouched
    c.close()


def test_failed_migration_rolls_back(tmp_path, monkeypatch):
    path = tmp_path / "digest.db"
    make_v1(path)
    before = counts(path)
    real_execute = sqlite3.Connection.execute

    class Boom(sqlite3.Connection):
        def execute(self, sql, *a):
            if "RENAME TO summaries" in sql:
                raise sqlite3.OperationalError("disk full (simulated)")
            return real_execute(self, sql, *a)

    real_connect = sqlite3.connect
    monkeypatch.setattr(sqlite3, "connect", lambda *a, **k: real_connect(*a, factory=Boom, **k))
    with pytest.raises(sqlite3.OperationalError):
        Database(path, default_topic="llm-evaluation")
    monkeypatch.undo()

    c = sqlite3.connect(path)
    assert "topic" not in {r[1] for r in c.execute("PRAGMA table_info(runs)")}  # nothing half-applied
    assert c.execute("PRAGMA user_version").fetchone()[0] == 0
    assert "topic_papers" not in {r[0] for r in c.execute("SELECT name FROM sqlite_master")}
    c.close()
    assert counts(path) == before
    with Database(path, default_topic="llm-evaluation") as db:  # and it migrates fine next time
        assert db.run_topic(1) == "llm-evaluation"


def test_dry_run_migrates_only_in_memory(tmp_path):
    path = tmp_path / "digest.db"
    make_v1(path)
    raw = path.read_bytes()
    with Database(path, in_memory_copy=True, default_topic="llm-evaluation") as db:
        assert db.run_topic(1) == "llm-evaluation"
    assert path.read_bytes() == raw and not (tmp_path / "digest.db.bak-schema1").exists()


def test_fresh_database_starts_at_the_current_schema(tmp_path):
    with Database(tmp_path / "new.db") as db:  # no data: no topic needed
        assert db.conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        assert "topic" in {r["name"] for r in db.conn.execute("PRAGMA table_info(runs)")}
        with pytest.raises(ValueError, match="no topic"):
            db.log_run(0, 0, 0, ["bm25"])  # and nothing silently lands in a "default" topic


# --- per-topic data ---
def test_a_paper_can_be_new_for_one_topic_and_seen_by_another(db, papers):
    db.save_papers(papers[:3], topic="llm-evaluation")
    assert len(db.filter_new(papers, topic="llm-evaluation")) == 7
    assert len(db.filter_new(papers, topic="ai-agents")) == 10  # same papers, still new here
    db.save_papers(papers[:2], topic="ai-agents")
    assert db.count_papers() == 3  # metadata stored once
    assert db.count_papers("llm-evaluation") == 3 and db.count_papers("ai-agents") == 2


def test_summaries_and_feedback_are_per_topic(db, papers):
    p = papers[0]
    db.save_papers([p], topic="a")
    db.save_papers([p], topic="b")
    db.put_summary(p.id, "m", SUMMARY, "pdf", topic="a")
    assert db.get_summary(p.id, "m", "a") is not None and db.get_summary(p.id, "m", "b") is None
    db.upsert_feedback(p.id, 1, 11, "2026-11-26T00:00:00Z", topic="a")
    db.upsert_feedback(p.id, -1, 22, "2026-11-26T00:00:00Z", topic="b")  # same paper, other topic
    assert db.get_feedback(p.id, "a")["rating"] == 1 and db.get_feedback(p.id, "b")["rating"] == -1
    db.delete_feedback(p.id, "a")
    assert db.get_feedback(p.id, "a") is None and db.get_feedback(p.id, "b") is not None


def test_close_previous_only_sees_the_same_topic(db):
    a1 = db.log_run(1, 1, 1, ["bm25"], topic="a")
    b1 = db.log_run(1, 1, 1, ["bm25"], topic="b")
    a2 = db.log_run(1, 1, 1, ["bm25"], topic="a")
    for run, issue in ((a1, 10), (b1, 11), (a2, 12)):
        db.save_digest_issue(run, issue)
    assert [r["issue_number"] for r in db.open_digests_before(a2)] == [10]  # not b's #11
    assert db.get_digest_issue(b1)["topic"] == "b"


def test_sent_in_digest_is_per_topic(db, papers):
    p = papers[0]
    db.save_papers([p], topic="a")
    db.save_papers([p], topic="b")
    run = db.log_run(1, 1, 1, ["bm25"], topic="a")
    db.save_delivery(run, p.id, 5, 55)
    sent = dict(db.conn.execute("SELECT topic, sent_in_digest FROM topic_papers WHERE paper_id = ?", (p.id,)))
    assert sent["a"] is not None and sent["b"] is None
    assert [r["topic"] for r in db.recent_deliveries("2000-01-01")] == ["a"]
    assert db.recent_deliveries("2000-01-01", topic="b") == []
