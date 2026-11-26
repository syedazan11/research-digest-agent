"""Reactions -> feedback, against the in-memory fake GitHub."""

import httpx

from digest.feedback import collect_feedback
from fake_github import FakeGitHub
from test_deliver import cfg, run, seed  # noqa: F401  (cfg is a fixture)

OWNER = "owner"  # fake_github.REPO = "owner/repo"


def delivered(db, cfg, papers):
    seed(db, cfg, papers)
    gh = FakeGitHub()
    run(db, cfg, gh)
    comments = {c["body"].split("paper=")[1].split(" ")[0]: c["id"] for c in gh.comments.values()}
    return gh, comments  # paper_id -> comment_id


def rating(db, paper_id, topic="llm-evaluation"):  # seed() logs runs under config.yaml's first topic
    row = db.get_feedback(paper_id, topic)
    return None if row is None else row["rating"]


def test_owner_reactions_become_feedback(db, cfg, papers):
    gh, cid = delivered(db, cfg, papers)
    a, b, c = (p.id for p in papers[:3])
    gh.react(cid[a], OWNER, "+1")
    gh.react(cid[b], OWNER, "-1")
    gh.react(cid[c], OWNER, "heart")  # not a rating

    r = collect_feedback(gh.client(), db)
    assert (r.added, r.comments_checked) == (2, 3)
    assert (rating(db, a), rating(db, b), rating(db, c)) == (1, -1, None)
    row = db.get_feedback(a, "llm-evaluation")
    assert row["source"] == "github" and row["comment_id"] == cid[a]


def test_only_the_owner_counts(db, cfg, papers):
    gh, cid = delivered(db, cfg, papers)
    a = papers[0].id
    gh.react(cid[a], "stranger", "-1")
    gh.react(cid[a], "Owner", "+1")  # GitHub logins are case-insensitive
    gh.react(cid[papers[1].id], "stranger", "+1")

    r = collect_feedback(gh.client(), db)
    assert rating(db, a) == 1 and rating(db, papers[1].id) is None
    assert r.ignored_users == {"stranger"}


def test_feedback_users_config(db, cfg, papers):
    gh, cid = delivered(db, cfg, papers)
    gh.react(cid[papers[0].id], "friend", "+1")
    collect_feedback(gh.client(), db, users=["owner", "friend"])
    assert rating(db, papers[0].id) == 1


def test_latest_reaction_wins(db, cfg, papers):
    gh, cid = delivered(db, cfg, papers)
    a = papers[0].id
    gh.react(cid[a], OWNER, "+1", at="2026-11-26T10:00:00Z")
    gh.react(cid[a], OWNER, "-1", at="2026-11-26T11:00:00Z")
    collect_feedback(gh.client(), db)
    assert rating(db, a) == -1


def test_changed_and_removed_reactions(db, cfg, papers):
    gh, cid = delivered(db, cfg, papers)
    a = papers[0].id
    gh.react(cid[a], OWNER, "+1")
    collect_feedback(gh.client(), db)

    gh.unreact(cid[a], OWNER)
    gh.react(cid[a], OWNER, "-1", at="2026-11-27T09:00:00Z")
    r = collect_feedback(gh.client(), db)
    assert r.changed == 1 and rating(db, a) == -1

    gh.unreact(cid[a], OWNER)
    r = collect_feedback(gh.client(), db)
    assert r.removed == 1 and rating(db, a) is None
    assert db.conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0] == 0


def test_polling_twice_is_idempotent(db, cfg, papers):
    gh, cid = delivered(db, cfg, papers)
    gh.react(cid[papers[0].id], OWNER, "+1")
    collect_feedback(gh.client(), db)
    r = collect_feedback(gh.client(), db)
    assert (r.added, r.changed, r.unchanged) == (0, 0, 1)
    assert db.conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0] == 1


def test_deleted_comment_keeps_existing_feedback(db, cfg, papers):
    gh, cid = delivered(db, cfg, papers)
    a = papers[0].id
    gh.react(cid[a], OWNER, "+1")
    collect_feedback(gh.client(), db)
    del gh.comments[cid[a]]  # someone deleted the comment
    collect_feedback(gh.client(), db)
    assert rating(db, a) == 1


def test_same_paper_in_two_digests_uses_newest_reaction(db, cfg, papers):
    gh, _ = delivered(db, cfg, papers)
    seed(db, cfg, papers)  # run 2 re-sends the same papers (--include-seen)
    run(db, cfg, gh)
    a = papers[0].id
    first, second = sorted(c["id"] for c in gh.comments.values() if f"paper={a} " in c["body"])
    gh.react(first, OWNER, "-1", at="2026-11-20T10:00:00Z")
    gh.react(second, OWNER, "+1", at="2026-11-27T10:00:00Z")
    collect_feedback(gh.client(), db)
    assert rating(db, a) == 1 and db.get_feedback(a, "llm-evaluation")["comment_id"] == second


def test_bad_token_is_reported_not_raised(db, cfg, papers):
    gh, cid = delivered(db, cfg, papers)
    gh.before = lambda req: httpx.Response(401, json={"message": "Bad credentials"})
    r = collect_feedback(gh.client(), db)
    assert "invalid or expired" in r.error


def test_old_deliveries_are_not_polled(db, cfg, papers):
    gh, cid = delivered(db, cfg, papers)
    db.conn.execute("UPDATE deliveries SET posted_at = '2026-01-01T00:00:00+00:00'")
    r = collect_feedback(gh.client(), db, days=30)
    assert r.comments_checked == 0



# --- Day 7: feedback per topic ---
def test_the_same_paper_rated_differently_in_two_topics(db, cfg, papers):
    gh = FakeGitHub()
    seed(db, cfg, papers, topic="llm-evaluation")
    seed(db, cfg, papers, topic="ai-agents")
    for rid in (1, 2):
        run_deliver = __import__("digest.deliver", fromlist=["deliver_run"]).deliver_run
        assert run_deliver(db, cfg, rid, gh.client(), sleep=lambda s: None).complete
    a = papers[0].id
    llm_eval_c, agents_c = sorted(c["id"] for c in gh.comments.values() if f"paper={a} " in c["body"])
    gh.react(llm_eval_c, OWNER, "+1")
    gh.react(agents_c, OWNER, "-1")

    collect_feedback(gh.client(), db)
    assert rating(db, a, "llm-evaluation") == 1 and rating(db, a, "ai-agents") == -1

    r = collect_feedback(gh.client(), db, topic="ai-agents")  # --topic: only that topic's comments
    assert r.comments_checked == 3
