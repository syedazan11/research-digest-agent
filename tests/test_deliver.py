"""GitHub delivery against the in-memory fake: content, resume after crashes, closing, errors."""

import json
import logging
from pathlib import Path

import httpx
import pytest

from digest.config import load_config
from digest.deliver import LABEL, build_content, deliver_run, preview
from fake_github import REPO, Crash, FakeGitHub

ROOT = Path(__file__).parent.parent
SUMMARY = {"problem": "Judges are unstable.", "method": "Vary scales.", "key_result": "Scores drop 7 points.",
           "why_it_matters": "Matters for @everyone using judges.", "limitations": "See #12 in the repo."}


@pytest.fixture
def cfg():
    return load_config(ROOT / "config.yaml")  # topic "LLM evaluation", rerank groq/openai/gpt-oss-120b


def seed(db, cfg, papers, n=3, run_id=None, topic=None):
    """Save a run that selected the first n sample papers, with summaries for all but the last."""
    topic = topic or cfg.topic.slug
    db.save_papers(papers, topic=topic)
    rid = db.log_run(len(papers), len(papers), 10, ["bm25"], notes="test", topic=topic)
    assert run_id is None or rid == run_id
    rows = [{"paper_id": p.id, "hybrid_rank": i + 2, "hybrid_score": 0.9 - i / 10, "llm_score": 9 - i,
             "reason": f"reason {i}", "final_rank": i + 1} for i, p in enumerate(papers[:n])]
    db.save_rankings(rid, rows)
    key = f"{cfg.rerank.provider}/{cfg.rerank.model}"
    for p in papers[: n - 1]:
        db.put_summary(p.id, key, json.dumps(SUMMARY), "pdf", topic=topic)
    return rid


def run(db, cfg, gh):
    return deliver_run(db, cfg, run_id=db.latest_undelivered_run(), gh=gh.client(), sleep=lambda s: None)


# --- content ---
def test_issue_and_comments_are_created(db, cfg, papers):
    rid = seed(db, cfg, papers)
    gh = FakeGitHub()
    result = run(db, cfg, gh)

    assert result.complete and result.posted == 3 and result.issue_number == 1
    issue = gh.issues[1]
    assert issue["title"].startswith("Research digest: LLM evaluation, 20") and issue["title"].endswith(f"(run {rid})")
    assert [x["name"] for x in issue["labels"]] == [LABEL, "topic:llm-evaluation"] and LABEL in gh.labels
    assert "**Scanned:** 10 papers from the last 7 days (10 new)" in issue["body"]
    assert "BM25 → top 10 → groq/openai/gpt-oss-120b rerank → top 3" in issue["body"]

    comments = gh.comments_on(1)
    assert len(comments) == 3
    first = comments[0]["body"]
    assert first.startswith(f"## 1. {papers[0].title}")
    assert "**Relevance:** 9/10 - reason 0" in first and "- **Key result:** Scores drop 7 points." in first
    assert first.rstrip().endswith(f"<!-- digest:run={rid} paper={papers[0].id} -->")
    assert "React 👍 or 👎" in first
    assert "*Summary unavailable. Abstract:*" in comments[2]["body"]  # third paper had no summary

    assert set(db.delivered(rid)) == {p.id for p in papers[:3]}
    assert db.get_digest_issue(rid)["completed_at"] is not None
    sent = db.conn.execute("SELECT COUNT(*) FROM topic_papers WHERE sent_in_digest IS NOT NULL").fetchone()[0]
    assert sent == 3


def test_mentions_and_issue_refs_are_escaped_in_comments(db, cfg, papers):
    seed(db, cfg, papers)
    gh = FakeGitHub()
    run(db, cfg, gh)
    body = gh.comments_on(1)[0]["body"]
    assert "@everyone" not in body and "#12" not in body
    assert "@\u2060everyone" in body and "\\#\u206012" in body


def test_comments_are_spaced_out(db, cfg, papers):
    seed(db, cfg, papers)
    gh = FakeGitHub()
    sleeps = []
    deliver_run(db, cfg, db.latest_undelivered_run(), gh.client(), sleep=sleeps.append)
    assert sleeps == [1.0, 1.0]  # between comments, not before the first


# --- idempotent resume ---
def test_second_run_posts_nothing(db, cfg, papers):
    rid = seed(db, cfg, papers)
    gh = FakeGitHub()
    run(db, cfg, gh)
    again = deliver_run(db, cfg, rid, gh.client(), sleep=lambda s: None)
    assert again.posted == 0 and again.skipped == 3
    assert len(gh.issues) == 1 and len(gh.comments) == 3


@pytest.mark.parametrize("when", ["before", "after"])
def test_resume_after_crash_never_double_posts(db, cfg, papers, when):
    """'before': died before GitHub saved comment 2. 'after': GitHub saved it, but we died before
    recording it, so only the hidden marker on GitHub knows it exists."""
    rid = seed(db, cfg, papers)
    gh = FakeGitHub()
    gh.crash_on_comment = (2, when)
    with pytest.raises(Crash):
        run(db, cfg, gh)
    assert len(db.delivered(rid)) == 1
    assert db.get_digest_issue(rid)["completed_at"] is None

    gh.crash_on_comment = None
    result = run(db, cfg, gh)  # resume
    bodies = [c["body"] for c in gh.comments_on(1)]
    assert len(gh.issues) == 1 and len(bodies) == 3  # exactly one comment per paper
    assert [b.split("\n")[0][:5] for b in bodies] == ["## 1.", "## 2.", "## 3."]
    assert result.recovered == (1 if when == "after" else 0)
    assert result.complete and set(db.delivered(rid)) == {p.id for p in papers[:3]}


def test_crash_after_issue_created_but_not_saved(db, cfg, papers, monkeypatch):
    rid = seed(db, cfg, papers)
    gh = FakeGitHub()
    real_save = db.save_digest_issue

    def crash(*a, **k):
        raise Crash("died right after GitHub created the issue")

    monkeypatch.setattr(db, "save_digest_issue", crash)
    with pytest.raises(Crash):
        run(db, cfg, gh)
    assert len(gh.issues) == 1 and db.get_digest_issue(rid) is None

    monkeypatch.setattr(db, "save_digest_issue", real_save)
    result = run(db, cfg, gh)
    assert len(gh.issues) == 1  # found by its marker, not opened again
    assert result.complete and len(gh.comments_on(1)) == 3


# --- closing the previous digest ---
def test_previous_digest_is_closed(db, cfg, papers):
    seed(db, cfg, papers)
    gh = FakeGitHub()
    run(db, cfg, gh)
    rid2 = seed(db, cfg, papers)
    result = run(db, cfg, gh)
    assert result.closed == [1]
    assert gh.issues[1]["state"] == "closed" and gh.issues[2]["state"] == "open"
    assert db.get_digest_issue(rid2)["closed_at"] is None


def test_close_previous_can_be_turned_off(db, cfg, papers):
    cfg.deliver.close_previous = False
    seed(db, cfg, papers)
    gh = FakeGitHub()
    run(db, cfg, gh)
    seed(db, cfg, papers)
    assert run(db, cfg, gh).closed == []
    assert gh.issues[1]["state"] == "open"


# --- errors never crash ---
def test_bad_token_gives_clear_error_not_crash(db, cfg, papers, caplog):
    seed(db, cfg, papers)
    gh = FakeGitHub()
    gh.before = lambda req: httpx.Response(401, json={"message": "Bad credentials"})
    with caplog.at_level(logging.ERROR):
        result = run(db, cfg, gh)
    assert not result.complete and "invalid or expired" in result.error
    assert "GitHub delivery failed" in caplog.text
    assert gh.issues == {}


def test_rate_limit_mid_delivery_stops_cleanly_with_resume_hint(db, cfg, papers, caplog):
    rid = seed(db, cfg, papers)
    gh = FakeGitHub()
    posts = {"n": 0}

    def limit_second_comment(req):
        if req.method == "POST" and req.url.path.endswith("/comments"):
            posts["n"] += 1
            if posts["n"] >= 2:
                return httpx.Response(403, headers={"retry-after": "3600"}, json={"message": "secondary rate limit"})
        return None

    gh.before = limit_second_comment
    with caplog.at_level(logging.ERROR):
        result = run(db, cfg, gh)
    assert result.error and len(db.delivered(rid)) == 1
    assert f"python -m digest.deliver --run {rid}" in caplog.text

    gh.before = None
    assert run(db, cfg, gh).complete and len(gh.comments_on(1)) == 3


def test_no_client_and_no_papers(db, cfg, papers):
    rid = seed(db, cfg, papers)
    assert deliver_run(db, cfg, rid, None).error == "no GitHub client"
    empty = db.log_run(0, 0, 0, ["bm25"])
    assert deliver_run(db, cfg, empty, FakeGitHub().client()).error == "no selected papers"


# --- preview ---
def test_preview_shows_issue_and_every_comment(db, cfg, papers):
    rid = seed(db, cfg, papers)
    text = preview(build_content(db, cfg, rid))
    assert "ISSUE  Research digest: LLM evaluation" in text and "[labels: digest, topic:llm-evaluation]" in text
    assert text.count("--- comment ") == 3


def test_latest_undelivered_run(db, cfg, papers):
    r1 = seed(db, cfg, papers)
    assert db.latest_undelivered_run() == r1
    run(db, cfg, FakeGitHub())
    assert db.latest_undelivered_run() is None


# --- Day 7: one issue per topic ---
def test_each_topic_gets_its_own_issue_and_closes_only_its_own_previous(db, cfg, papers):
    gh = FakeGitHub()
    a1 = seed(db, cfg, papers, topic="llm-evaluation")
    b1 = seed(db, cfg, papers, topic="ai-agents")
    for rid in (a1, b1):
        assert deliver_run(db, cfg, rid, gh.client(), sleep=lambda s: None).complete
    a2 = seed(db, cfg, papers, topic="llm-evaluation")
    result = deliver_run(db, cfg, a2, gh.client(), sleep=lambda s: None)

    by_title = {i["title"].split(",")[0]: i for i in gh.issues.values()}
    assert set(by_title) == {"Research digest: LLM evaluation", "Research digest: AI agents"}
    agents = gh.issues[2]
    assert agents["title"].startswith("Research digest: AI agents")
    assert [x["name"] for x in agents["labels"]] == ["digest", "topic:ai-agents"]
    assert {"digest", "topic:llm-evaluation", "topic:ai-agents"} <= gh.labels
    assert result.closed == [1]  # llm-evaluation's previous issue...
    assert gh.issues[1]["state"] == "closed" and gh.issues[2]["state"] == "open"  # ...not ai-agents'
    assert "Weekly arXiv digest for **AI agents**" in agents["body"]


def test_a_topic_removed_from_config_still_delivers(db, cfg, papers):
    rid = seed(db, cfg, papers, topic="retired-topic")
    gh = FakeGitHub()
    assert deliver_run(db, cfg, rid, gh.client(), sleep=lambda s: None).complete
    assert gh.issues[1]["title"].startswith("Research digest: retired-topic")
