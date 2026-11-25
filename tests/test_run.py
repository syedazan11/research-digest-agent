"""End-to-end pipeline on the sample papers. make_llm is patched, so no real API call is possible."""

import json

import pytest

import digest.run as run
from conftest import FIXTURES, FakeLLM, score_by_title
from digest.db import Database
from digest.pdf import PdfFetcher, Throttle

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
    monkeypatch.setattr(run, "PdfFetcher", OfflineFetcher)  # never download a real PDF
    monkeypatch.setattr(run, "make_github", lambda: None)  # never post to a real repo
    return cfg


class OfflineFetcher(PdfFetcher):
    def __init__(self, delay_seconds=3.0, **kw):
        super().__init__(delay_seconds, get=self._no_network, throttle=Throttle(0))

    @staticmethod
    def _no_network(url):
        raise ConnectionError("tests are offline")


SUMMARY = {"problem": "P.", "method": "M.", "key_result": "Citations raise trust by 12%.",
           "why_it_matters": "W.", "limitations": "Not stated in the provided text."}


def rerank_and_summarize(scores: dict[str, float]):
    """FakeLLM reply function answering both stages: summary prompts get a summary, rerank prompts scores."""
    score = score_by_title(scores)

    def fn(user):
        return json.dumps(SUMMARY) if user.startswith("Summarize this paper") else score(user)

    return fn


def args(cfg, *extra):
    return ["--config", str(cfg), "--sample", str(FIXTURES / "sample_papers.json"), "--no-embeddings", *extra]


def test_run_with_rerank_saves_scores(setup, monkeypatch, capsys):
    llm = FakeLLM(default=score_by_title({"Grounded Answer": 10, "Hallucinations": 9}))
    monkeypatch.setattr(run, "make_llm", lambda provider, model: llm)

    assert run.main(args(setup)) == 0

    out = capsys.readouterr().out
    assert "Rerank (rerank: fake/fake-1 prompt v1, 10/10 scored)" in out
    assert "fake reason" in out

    [cand_file] = (setup.parent / "data" / "candidates").glob("*.json")
    assert cand_file.name.endswith("_run1.json")
    cands = json.loads(cand_file.read_text(encoding="utf-8"))
    assert [c["paper"]["id"] for c in cands[:2]] == ["sample-004", "sample-002"]
    assert cands[0]["final_rank"] == 1 and cands[0]["llm_score"] == 10 and cands[0]["llm_reason"]
    assert cands[3]["final_rank"] is None

    with Database(setup.parent / "data" / "test.db") as db:
        rows = db.get_rankings(1)
        assert len(rows) == 10 and sum(r["selected"] for r in rows) == 3
        note = db.conn.execute("SELECT notes FROM runs WHERE id = 1").fetchone()[0]
        assert note.startswith("rerank: fake")


def test_no_rerank_no_summarize_never_builds_an_llm(setup, monkeypatch, capsys):
    def boom(*a, **k):
        raise AssertionError("make_llm must not be called when no stage needs an LLM")

    monkeypatch.setattr(run, "make_llm", boom)
    assert run.main(args(setup, "--no-rerank", "--no-summarize", "--dry-run")) == 0
    out = capsys.readouterr().out
    assert "Rerank (rerank off)" in out and "summaries off" in out


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
    # U+2011 is normalized to "-" (prompting.normalize_text); the arrow isn't, and becomes "?"
    assert "LLM-as-a-judge ? relevant" in console.buffer.getvalue().decode("cp1252")
    with Database(setup.parent / "data" / "test.db") as db:
        assert len(db.get_rankings(1)) == 10  # saved


def files_under(root):
    return sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())


def test_dry_run_writes_nothing_on_a_fresh_setup(setup, monkeypatch):
    monkeypatch.setattr(run, "make_llm", lambda provider, model: FakeLLM(default=score_by_title({})))
    before = files_under(setup.parent)
    assert run.main(args(setup, "--dry-run")) == 0
    assert files_under(setup.parent) == before  # no DB file, no candidates file, nothing


def test_dry_run_leaves_an_existing_db_untouched(setup, monkeypatch):
    monkeypatch.setattr(run, "make_llm", lambda provider, model: FakeLLM(default=score_by_title({})))
    assert run.main(args(setup)) == 0  # a real run: DB + 1 candidates file
    db_file = setup.parent / "data" / "test.db"
    db_bytes = db_file.read_bytes()
    before = files_under(setup.parent)

    assert run.main(args(setup, "--dry-run", "--include-seen")) == 0
    assert files_under(setup.parent) == before
    assert db_file.read_bytes() == db_bytes


def test_runs_on_the_same_day_never_overwrite(setup, monkeypatch):
    monkeypatch.setattr(run, "make_llm", lambda provider, model: FakeLLM(default=score_by_title({})))
    assert run.main(args(setup)) == 0
    assert run.main(args(setup, "--include-seen")) == 0
    names = [p.name for p in sorted((setup.parent / "data" / "candidates").glob("*.json"))]
    assert len(names) == 2
    assert names[0].endswith("_run1.json") and names[1].endswith("_run2.json")


# --- Day 3: summaries + digest ---
def test_full_run_writes_markdown_digest(setup, monkeypatch, caplog):
    llm = FakeLLM(default=rerank_and_summarize({"Grounded Answer": 10, "Hallucinations": 9}))
    monkeypatch.setattr(run, "make_llm", lambda provider, model: llm)
    with caplog.at_level("INFO"):
        assert run.main(args(setup)) == 0

    [digest_file] = (setup.parent / "data" / "digests").glob("*.md")
    assert digest_file.name.endswith("_run1.md")
    md = digest_file.read_text(encoding="utf-8")
    assert md.startswith("# Research digest: RAG evaluation\n")
    assert "run 1 · 3 papers · ranked by fake/fake-1" in md
    assert "## 1. Grounded Answer Generation: Do Citations Help Users Trust Model Outputs?" in md
    assert "**Relevance:** 10/10 - fake reason" in md
    assert md.count("- **Key result:** Citations raise trust by 12%.") == 3
    # sample papers have no PDF link, so every summary is abstract-based and says so
    assert md.count("*Summary based on the abstract only (download_failed).*") == 3

    assert "LLM calls this run: 5 (rerank 2, summarize 3)" in caplog.text
    assert "Timing: fetch" in caplog.text and "summarize" in caplog.text
    with Database(setup.parent / "data" / "test.db") as db:
        assert db.conn.execute("SELECT COUNT(*) FROM summaries").fetchone()[0] == 3
        note = db.conn.execute("SELECT notes FROM runs WHERE id = 1").fetchone()[0]
        assert "summaries: 3/3 (0 from PDF, 3 abstract-only, 0 cached)" in note


def test_second_run_uses_summary_cache(setup, monkeypatch, caplog):
    # a fresh client per run, like real runs (the call counter is per client)
    monkeypatch.setattr(run, "make_llm", lambda provider, model: FakeLLM(default=rerank_and_summarize({})))
    assert run.main(args(setup)) == 0
    caplog.clear()
    with caplog.at_level("INFO"):
        assert run.main(args(setup, "--include-seen")) == 0
    assert "(rerank 2, summarize 0)" in caplog.text
    assert len(list((setup.parent / "data" / "digests").glob("*.md"))) == 2  # _run1.md and _run2.md


def test_no_summarize_still_writes_a_digest_with_abstracts(setup, monkeypatch):
    llm = FakeLLM(default=score_by_title({}))  # would fail on a summary prompt
    monkeypatch.setattr(run, "make_llm", lambda provider, model: llm)
    assert run.main(args(setup, "--no-summarize")) == 0
    assert llm.calls == 2  # rerank only
    [digest_file] = (setup.parent / "data" / "digests").glob("*.md")
    md = digest_file.read_text(encoding="utf-8")
    assert md.count("*Summary unavailable. Abstract:*") == 3 and "**Key result:**" not in md


def test_dry_run_writes_no_digest(setup, monkeypatch):
    monkeypatch.setattr(run, "make_llm", lambda provider, model: FakeLLM(default=rerank_and_summarize({})))
    assert run.main(args(setup, "--dry-run")) == 0
    assert not (setup.parent / "data").exists()


# --- Day 4: GitHub delivery ---
def test_run_delivers_digest_to_github(setup, monkeypatch, caplog):
    from fake_github import FakeGitHub

    gh = FakeGitHub()
    monkeypatch.setattr(run, "make_llm", lambda provider, model: FakeLLM(default=rerank_and_summarize({"Grounded": 10})))
    monkeypatch.setattr(run, "make_github", lambda: gh.client())
    with caplog.at_level("INFO"):
        assert run.main(args(setup)) == 0

    assert len(gh.issues) == 1 and gh.issues[1]["title"].endswith("(run 1)")
    comments = gh.comments_on(1)
    assert len(comments) == 3 and comments[0]["body"].startswith("## 1. Grounded Answer Generation")
    assert "Delivered run 1 to issue #1: 3 comments posted" in caplog.text
    assert "deliver" in caplog.text.split("Timing:")[1]

    # the .md digest and the issue comment are rendered by the same code
    [digest_file] = (setup.parent / "data" / "digests").glob("*.md")
    first_paper_md = digest_file.read_text(encoding="utf-8").split("---\n\n")[1].strip()
    assert comments[0]["body"].startswith(first_paper_md)


def test_no_deliver_flag(setup, monkeypatch, caplog):
    def boom():
        raise AssertionError("make_github must not be called with --no-deliver")

    monkeypatch.setattr(run, "make_llm", lambda provider, model: FakeLLM(default=rerank_and_summarize({})))
    monkeypatch.setattr(run, "make_github", boom)
    with caplog.at_level("INFO"):
        assert run.main(args(setup, "--no-deliver")) == 0
    assert "Delivery off (--no-deliver)" in caplog.text


def test_dry_run_previews_issue_without_calling_github(setup, monkeypatch, capsys):
    def boom():
        raise AssertionError("--dry-run must not call GitHub")

    monkeypatch.setattr(run, "make_llm", lambda provider, model: FakeLLM(default=rerank_and_summarize({})))
    monkeypatch.setattr(run, "make_github", boom)
    assert run.main(args(setup, "--dry-run")) == 0
    out = capsys.readouterr().out
    assert "GitHub issue preview (dry run: nothing posted)" in out
    assert "ISSUE  Research digest: RAG evaluation" in out and out.count("--- comment ") == 3
    assert not (setup.parent / "data").exists()  # still writes nothing


def test_github_failure_never_fails_the_run(setup, monkeypatch, caplog):
    import httpx
    from fake_github import FakeGitHub

    gh = FakeGitHub()
    gh.before = lambda req: httpx.Response(401, json={"message": "Bad credentials"})
    monkeypatch.setattr(run, "make_llm", lambda provider, model: FakeLLM(default=rerank_and_summarize({})))
    monkeypatch.setattr(run, "make_github", lambda: gh.client())
    with caplog.at_level("ERROR"):
        assert run.main(args(setup)) == 0
    assert "GITHUB_TOKEN is invalid or expired" in caplog.text
    assert len(list((setup.parent / "data" / "digests").glob("*.md"))) == 1  # digest still saved


# --- Day 5: --strict for CI ---
def test_strict_clean_run_exits_0(setup, monkeypatch):
    from fake_github import FakeGitHub

    gh = FakeGitHub()
    monkeypatch.setattr(run, "make_llm", lambda provider, model: FakeLLM(default=rerank_and_summarize({})))
    monkeypatch.setattr(run, "make_github", lambda: gh.client())
    assert run.main(args(setup, "--strict")) == 0


def test_strict_github_failure_exits_1_but_saves_everything(setup, monkeypatch, caplog):
    import httpx
    from fake_github import FakeGitHub

    gh = FakeGitHub()
    gh.before = lambda req: httpx.Response(401, json={"message": "Bad credentials"})
    monkeypatch.setattr(run, "make_llm", lambda provider, model: FakeLLM(default=rerank_and_summarize({})))
    monkeypatch.setattr(run, "make_github", lambda: gh.client())
    with caplog.at_level("ERROR"):
        assert run.main(args(setup, "--strict")) == 1
    assert "Degraded run: GitHub delivery failed" in caplog.text
    assert len(list((setup.parent / "data" / "digests").glob("*.md"))) == 1
    with Database(setup.parent / "data" / "test.db") as db:
        assert db.get_run(1) is not None and len(db.selected_rankings(1)) == 3


def test_strict_missing_llm_key_exits_1(setup, monkeypatch, caplog):
    from fake_github import FakeGitHub

    monkeypatch.setattr(run, "make_llm", lambda provider, model: None)
    monkeypatch.setattr(run, "make_github", lambda: FakeGitHub().client())
    with caplog.at_level("ERROR"):
        assert run.main(args(setup, "--strict")) == 1
    assert "LLM rerank fell back" in caplog.text and "no paper got a summary" in caplog.text


def test_degraded_run_without_strict_still_exits_0(setup, monkeypatch, caplog):
    monkeypatch.setattr(run, "make_llm", lambda provider, model: None)  # make_github is None too
    with caplog.at_level("WARNING"):
        assert run.main(args(setup)) == 0
    assert "Degraded run: GitHub delivery skipped" in caplog.text


# --- Day 6: feedback retriever in the pipeline ---
def test_run_uses_feedback_retriever_when_weighted(setup, monkeypatch, caplog):
    from digest.labels import LabelStore
    from digest.fetch import load_sample

    cfg_text = setup.read_text(encoding="utf-8").replace(
        "ranking:\n  top_k: 10", "ranking:\n  top_k: 10\n  weights:\n    bm25: 0.6\n    feedback: 0.4")
    setup.write_text(cfg_text + "  labels_path: data/labels.db\n", encoding="utf-8")
    sample = load_sample(FIXTURES / "sample_papers.json")
    with LabelStore(setup.parent / "data" / "labels.db") as s:
        s.add_to_pool(sample)
        s.set_label("sample-002", True)
        s.set_label("sample-005", False)
    monkeypatch.setattr(run, "make_llm", lambda provider, model: None)
    with caplog.at_level("INFO"):
        assert run.main(args(setup, "--dry-run", "--no-deliver")) == 0
    assert "Feedback retriever: 1 liked, 1 disliked (0 reactions, 2 labels), weight 0.40" in caplog.text
    assert "Retrievers: bm25, feedback" in caplog.text
