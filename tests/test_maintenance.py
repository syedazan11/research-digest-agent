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

    assert commit_message(path, since_run=before) == f"digest: test-topic run {run} (5 papers, issue #7)"
    assert commit_message(path, since_run=run) == "digest: feedback only (1 ratings)"


def test_commit_message_without_issue(tmp_path):
    path = tmp_path / "d.db"
    with Database(path, default_topic="test-topic") as db:
        db.log_run(0, 0, 0, ["bm25"])
    assert commit_message(path, since_run=0) == "digest: test-topic run 1 (0 papers)"


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
    assert out[0] == "1" and out[1] == "digest: test-topic run 1 (0 papers)" and out[2].startswith("VACUUM d.db")



def test_commit_message_lists_every_topics_run(tmp_path, papers):
    path = tmp_path / "d.db"
    with Database(path, default_topic="llm-evaluation") as db:
        before = db.log_run(1, 1, 1, ["bm25"])
        for topic, issue in (("llm-evaluation", 9), ("ai-agents", 10)):
            db.save_papers(papers[:5], topic)
            run = db.log_run(10, 10, 10, ["bm25"], topic=topic)
            db.save_rankings(run, [{"paper_id": p.id, "hybrid_rank": i, "hybrid_score": 0.5, "final_rank": i}
                                   for i, p in enumerate(papers[:5], 1)])
            db.save_digest_issue(run, issue)
    assert commit_message(path, since_run=before) == (
        "digest: llm-evaluation run 2 (5 papers, issue #9); ai-agents run 3 (5 papers, issue #10)")


# --- reset (for a copy made from the template) ---
import pytest  # noqa: E402

from digest.config import load_config  # noqa: E402
from digest.labels import LabelStore  # noqa: E402
from digest.maintenance import PROTECTED_REPO, ResetRefused, current_repo, reset_data  # noqa: E402


def copy_with_data(tmp_path, papers):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("topics:\n  - name: My topic\n    keywords: [k]\n"
                        "database:\n  path: data/digest.db\n  labels_path: data/labels.db\n", encoding="utf-8")
    cfg = load_config(cfg_path)
    with Database(cfg.db_path, default_topic="my-topic") as db:
        db.save_papers(papers)
        db.log_run(10, 10, 10, ["bm25"])
    with LabelStore(cfg.labels_path, default_topic="my-topic") as s:
        s.add_to_pool(papers)
        s.set_label(papers[0].id, True)
    (tmp_path / "data" / "evals").mkdir()
    (tmp_path / "data" / "evals" / "2026-11-26.md").write_text("someone else's results", encoding="utf-8")
    return cfg


def test_reset_empties_both_databases_after_backing_them_up(tmp_path, papers):
    cfg = copy_with_data(tmp_path, papers)
    done = reset_data(cfg, yes=True, repo="someone/their-copy")
    with Database(cfg.db_path, default_topic="my-topic") as db:
        assert db.count_papers() == 0 and db.conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0
    with LabelStore(cfg.labels_path, default_topic="my-topic") as s:
        assert s.stats()["pool"] == 0
    backups = sorted(p.name for p in (tmp_path / "data").glob("*.bak-reset-*"))
    assert len(backups) == 2 and backups[0].startswith("digest.db.bak-reset-")
    with Database(tmp_path / "data" / backups[0], default_topic="my-topic") as old:
        assert old.count_papers() == 10  # the backup really holds the old data
    assert not (tmp_path / "data" / "evals").exists()
    assert any("removed evals/" in line for line in done)


def test_reset_needs_yes(tmp_path, papers):
    cfg = copy_with_data(tmp_path, papers)
    with pytest.raises(ResetRefused, match="--yes"):
        reset_data(cfg, yes=False, repo="someone/their-copy")
    with Database(cfg.db_path, default_topic="my-topic") as db:
        assert db.count_papers() == 10  # untouched


@pytest.mark.parametrize("repo", [PROTECTED_REPO, "SyedAzan11/Research-Digest-Agent"])
def test_reset_refuses_on_the_original_repo_even_with_yes(tmp_path, papers, repo):
    cfg = copy_with_data(tmp_path, papers)
    with pytest.raises(ResetRefused, match="original project"):
        reset_data(cfg, yes=True, repo=repo)
    assert not list((tmp_path / "data").glob("*.bak-reset-*"))


def test_current_repo_from_env_or_git_remote(tmp_path, monkeypatch):
    import subprocess

    monkeypatch.setenv("GITHUB_REPOSITORY", "someone/their-copy")
    assert current_repo(tmp_path) == "someone/their-copy"
    monkeypatch.delenv("GITHUB_REPOSITORY")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    for url in ("https://github.com/someone/their-copy.git", "git@github.com:someone/their-copy.git"):
        subprocess.run(["git", "remote", "remove", "origin"], cwd=tmp_path, capture_output=True)
        subprocess.run(["git", "remote", "add", "origin", url], cwd=tmp_path, check=True)
        assert current_repo(tmp_path) == "someone/their-copy"


def test_reset_cli(tmp_path, papers, monkeypatch, capsys):
    cfg = copy_with_data(tmp_path, papers)
    monkeypatch.setenv("GITHUB_REPOSITORY", PROTECTED_REPO)
    assert main(["--config", str(tmp_path / "config.yaml"), "reset", "--yes"]) == 1
    assert "Reset refused" in capsys.readouterr().err
    monkeypatch.setenv("GITHUB_REPOSITORY", "someone/their-copy")
    assert main(["--config", str(tmp_path / "config.yaml"), "reset"]) == 1  # no --yes
    assert main(["--config", str(tmp_path / "config.yaml"), "reset", "--yes"]) == 0
    assert "Reset done" in capsys.readouterr().out
