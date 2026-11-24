"""End-to-end pipeline on the sample papers. make_llm is patched, so no real API call is possible."""

import json

import pytest

import digest.run as run
from conftest import FIXTURES, FakeLLM, score_by_title
from digest.db import Database

CONFIG = """
topic:
  name: RAG evaluation
  description: Methods, benchmarks and metrics for evaluating retrieval-augmented generation systems.
  keywords: [RAG evaluation, retrieval-augmented generation, faithfulness, hallucination]
  categories: [cs.CL]
ranking:
  top_k: 10
embeddings:
  provider: none
rerank:
  top_n: 3
  batch_size: 5
database:
  path: data/test.db
"""


@pytest.fixture
def setup(tmp_path, monkeypatch):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(CONFIG, encoding="utf-8")
    monkeypatch.setattr(run, "load_dotenv", lambda *a, **k: None)  # never load the real .env
    return cfg


def args(cfg, *extra):
    return ["--config", str(cfg), "--sample", str(FIXTURES / "sample_papers.json"), "--no-embeddings", *extra]


def test_run_with_rerank_saves_scores(setup, monkeypatch, capsys):
    llm = FakeLLM(default=score_by_title({"Grounded Answer": 10, "Hallucinations": 9}))
    monkeypatch.setattr(run, "make_llm", lambda provider, model: llm)

    assert run.main(args(setup)) == 0

    out = capsys.readouterr().out
    assert "Rerank (rerank: fake/fake-1, 10/10 scored)" in out
    assert "fake reason" in out

    cand_file = next((setup.parent / "data" / "candidates").glob("*.json"))
    cands = json.loads(cand_file.read_text(encoding="utf-8"))
    assert [c["paper"]["id"] for c in cands[:2]] == ["sample-004", "sample-002"]
    assert cands[0]["final_rank"] == 1 and cands[0]["llm_score"] == 10 and cands[0]["llm_reason"]
    assert cands[3]["final_rank"] is None

    with Database(setup.parent / "data" / "test.db") as db:
        rows = db.get_rankings(1)
        assert len(rows) == 10 and sum(r["selected"] for r in rows) == 3
        note = db.conn.execute("SELECT notes FROM runs WHERE id = 1").fetchone()[0]
        assert note.startswith("rerank: fake")


def test_no_rerank_flag_never_builds_an_llm(setup, monkeypatch, capsys):
    def boom(*a, **k):
        raise AssertionError("make_llm must not be called with --no-rerank")

    monkeypatch.setattr(run, "make_llm", boom)
    assert run.main(args(setup, "--no-rerank", "--dry-run")) == 0
    assert "Rerank (rerank off)" in capsys.readouterr().out


def test_missing_key_falls_back_to_hybrid(setup, monkeypatch, capsys):
    monkeypatch.setattr(run, "make_llm", lambda provider, model: None)
    assert run.main(args(setup, "--dry-run")) == 0
    assert "rerank skipped: no groq client" in capsys.readouterr().out


def test_unicode_reason_does_not_crash_a_cp1252_console(setup, monkeypatch):
    """Regression: a real Groq reason contained U+2011 (non-breaking hyphen) and crashed print()
    on a Windows console, before anything was saved."""
    import io
    import sys

    from conftest import reply

    console = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")  # what Windows gives you
    monkeypatch.setattr(sys, "stdout", console)
    llm = FakeLLM(default=lambda user: reply({"P1": 9}, reason="LLM\u2011as\u2011a\u2011judge \u2192 relevant"))
    monkeypatch.setattr(run, "make_llm", lambda provider, model: llm)

    assert run.main(args(setup)) == 0
    console.flush()
    assert "LLM?as?a?judge ? relevant" in console.buffer.getvalue().decode("cp1252")
    with Database(setup.parent / "data" / "test.db") as db:
        assert len(db.get_rankings(1)) == 10  # saved
