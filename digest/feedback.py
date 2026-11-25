"""Read 👍/👎 reactions on recent digest comments into the feedback table.

Rules:
- Only reactions from the repo owner count (or deliver.feedback_users). The repo is public,
  so anyone could react; strangers must not steer the ranking.
- +1 -> 1, -1 -> -1. Other reactions (heart, laugh, ...) are ignored.
- Latest state wins: if both 👍 and 👎 are there, the newer one counts. A paper sent in two
  digests (--include-seen) counts its newest reaction across both comments.
- A removed reaction removes the feedback, so the table always mirrors GitHub.
- Papers whose comments can't be read (deleted, network) keep their existing feedback.

Usage:  python -m digest.feedback
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from digest.db import Database
from digest.github import GitHubAuthError, GitHubClient, GitHubError, GitHubNotFound

log = logging.getLogger(__name__)

RATINGS = {"+1": 1, "-1": -1}


@dataclass
class FeedbackResult:
    comments_checked: int = 0
    added: int = 0
    changed: int = 0
    removed: int = 0
    unchanged: int = 0
    ignored_users: set[str] = field(default_factory=set)  # reacted, but don't count
    error: str = ""


def collect_feedback(gh: GitHubClient, db: Database, days: int = 30,
                     users: list[str] | None = None) -> FeedbackResult:
    result = FeedbackResult()
    allowed = {u.lower() for u in (users or [gh.owner])}
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(timespec="seconds")

    comments_by_paper: dict[str, list[int]] = {}
    for row in db.recent_deliveries(since):
        comments_by_paper.setdefault(row["paper_id"], []).append(row["comment_id"])

    try:
        for paper_id, comment_ids in comments_by_paper.items():
            latest: tuple[str, int, int] | None = None  # (reacted_at, rating, comment_id)
            read_any = False
            for cid in comment_ids:
                try:
                    reactions = gh.list_reactions(cid)
                except GitHubNotFound:
                    log.warning("Comment %d (paper %s) is gone from GitHub: skipped", cid, paper_id)
                    continue
                read_any = True
                result.comments_checked += 1
                for r in reactions:
                    login = (r.get("user") or {}).get("login", "")
                    rating = RATINGS.get(r.get("content", ""))
                    if rating is None:
                        continue
                    if login.lower() not in allowed:
                        result.ignored_users.add(login)
                        continue
                    candidate = (r.get("created_at", ""), rating, cid)
                    if latest is None or candidate[0] > latest[0]:  # ISO timestamps sort as text
                        latest = candidate

            if not read_any:
                continue  # can't see GitHub's state: keep what we have
            old = db.get_feedback(paper_id)
            if latest is None:
                if db.delete_feedback(paper_id):
                    result.removed += 1
                    log.info("Feedback removed for %s (reaction taken back)", paper_id)
                continue
            if db.upsert_feedback(paper_id, latest[1], latest[2], latest[0]):
                if old is None:
                    result.added += 1
                else:
                    result.changed += 1
                log.info("Feedback for %s: %s", paper_id, "👍" if latest[1] > 0 else "👎")
            else:
                result.unchanged += 1
    except GitHubAuthError as e:
        log.error("Reading feedback failed: %s", e)
        result.error = str(e)
    except GitHubError as e:
        log.error("Reading feedback stopped (%s); run it again later", e)
        result.error = str(e)

    if result.ignored_users:
        log.info("Ignored reactions from %d other account(s): only %s count",
                 len(result.ignored_users), ", ".join(sorted(allowed)))
    return result


def main(argv: list[str] | None = None) -> int:
    from dotenv import load_dotenv

    from digest.config import load_config
    from digest.github import make_github

    ap = argparse.ArgumentParser(description="Read 👍/👎 reactions on digest comments into the DB")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    load_dotenv()
    cfg = load_config(args.config)

    gh = make_github()
    if gh is None:
        return 1
    try:
        with Database(cfg.db_path) as db:
            r = collect_feedback(gh, db, cfg.deliver.feedback_days, cfg.deliver.feedback_users)
            total = db.conn.execute("SELECT COUNT(*) FROM feedback WHERE source = 'github'").fetchone()[0]
    finally:
        gh.close()
    log.info("Feedback: %d comments checked, %d added, %d changed, %d removed, %d unchanged (%d in DB), "
             "%d GitHub requests", r.comments_checked, r.added, r.changed, r.removed, r.unchanged, total, gh.requests)
    return 1 if r.error else 0


if __name__ == "__main__":
    sys.exit(main())
