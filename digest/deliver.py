"""Stage 5: deliver a run's digest as a GitHub issue: one issue, one comment per paper.

Why GitHub Issues? Free, works everywhere (Telegram is blocked in Pakistan), notifications come
by email and the mobile app, and reactions give 👍/👎 feedback with no bot server to run.

Idempotency, in two layers:
1. The DB (fast path): the issue is saved the moment it's created, and each comment the moment
   it's posted. A re-run skips what's recorded.
2. Hidden markers on GitHub (source of truth): a crash can land *after* GitHub saved a post but
   *before* we recorded it. On resume, existing issues/comments are matched by their marker
   (<!-- digest:run=4 ... -->), so nothing is ever posted twice.

Usage:
    python -m digest.deliver            # finish the newest run that isn't fully delivered
    python -m digest.deliver --run 4    # deliver (or resume) run 4
    python -m digest.deliver --dry-run  # print the issue instead of posting it
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Callable

from digest.config import Config
from digest.db import Database
from digest.github import GitHubAuthError, GitHubClient, GitHubError, GitHubNotFound
from digest.prompting import parse_json_model
from digest.render import md, render_paper
from digest.rerank import RerankedPaper
from digest.retrieval import RankedPaper
from digest.summarize import PaperSummary, Summary

log = logging.getLogger(__name__)

LABEL = "digest"
_RUN_MARKER = "<!-- digest:run={run_id} -->"
_PAPER_MARKER = "<!-- digest:run={run_id} paper={paper_id} -->"
_PAPER_MARKER_RE = re.compile(r"<!-- digest:run=(\d+) paper=(\S+) -->")
RETRIEVER_NAMES = {"bm25": "BM25", "embedding": "Gemini embeddings"}


@dataclass
class DigestContent:
    run_id: int
    title: str
    body: str
    comments: list[tuple[str, str]]  # (paper_id, comment body), in final-rank order


@dataclass
class DeliveryResult:
    run_id: int
    issue_number: int | None = None
    issue_url: str = ""
    posted: int = 0
    recovered: int = 0  # comments found on GitHub (marker) that the DB didn't know about
    skipped: int = 0  # already recorded in the DB
    closed: list[int] = field(default_factory=list)
    error: str = ""

    @property
    def complete(self) -> bool:
        return not self.error and self.issue_number is not None


# --- building the digest from the DB ---
def model_key(cfg: Config) -> str:
    return f"{cfg.rerank.provider}/{cfg.rerank.model}"


def load_entries(db: Database, cfg: Config, run_id: int) -> list[tuple[RerankedPaper, PaperSummary]]:
    """The selected papers of a run with their scores and cached summaries, in final order."""
    entries = []
    for r in db.selected_rankings(run_id):
        paper = db.get_paper(r["paper_id"])
        if paper is None:
            log.warning("Run %d: paper %s missing from the DB, left out", run_id, r["paper_id"])
            continue
        item = RerankedPaper(RankedPaper(paper, r["hybrid_score"]), hybrid_rank=r["hybrid_rank"],
                             llm_score=r["llm_score"], reason=r["reason"] or "", final_rank=r["final_rank"])
        row = db.get_summary(paper.id, model_key(cfg))
        summary = None
        if row is not None:
            try:
                summary = parse_json_model(Summary, row["summary"])
            except ValueError:
                summary = None
        ps = PaperSummary(paper, summary, row["source"] if row else "abstract",
                          row["fallback_reason"] if row else None)
        entries.append((item, ps))
    return entries


def build_content(db: Database, cfg: Config, run_id: int) -> DigestContent:
    run = db.get_run(run_id)
    if run is None:
        raise ValueError(f"run {run_id} not found")
    entries = load_entries(db, cfg, run_id)
    date = run["started_at"][:10]
    retrievers = " + ".join(RETRIEVER_NAMES.get(n, n) for n in json.loads(run["retrievers"])) or "hybrid"
    if any(item.llm_score is not None for item, _ in entries):
        ranking = f"{retrievers} → top {run['n_ranked']} → {md(model_key(cfg))} rerank → top {len(entries)}"
    else:
        ranking = f"{retrievers} → top {len(entries)} (no LLM rerank this run)"

    body = "\n".join([
        f"Weekly arXiv digest for **{md(cfg.topic.name)}**.",
        "",
        f"- **Scanned:** {run['n_fetched']} papers from the last {cfg.fetch.days_back} days "
        f"({run['n_new']} new)",
        f"- **Ranked:** {ranking}",
        "- **Feedback:** react 👍 or 👎 on each paper's comment below. Only the repo owner's reactions count.",
        "",
        _RUN_MARKER.format(run_id=run_id),
    ])
    comments = []
    for item, ps in entries:
        lines = render_paper(item.final_rank or 0, item, ps)
        lines += ["React 👍 or 👎 to rate this paper.", "",
                  _PAPER_MARKER.format(run_id=run_id, paper_id=item.ranked.paper.id)]
        comments.append((item.ranked.paper.id, "\n".join(lines)))
    title = f"Research digest: {cfg.topic.name}, {date} (run {run_id})"
    return DigestContent(run_id, title, body, comments)


def preview(content: DigestContent) -> str:
    """What --dry-run prints instead of calling GitHub."""
    bar = "=" * 100
    parts = [bar, f"ISSUE  {content.title}   [label: {LABEL}]", bar, content.body]
    for i, (_, body) in enumerate(content.comments, 1):
        parts += ["", f"--- comment {i}/{len(content.comments)} " + "-" * 80, body]
    return "\n".join(parts)


# --- posting ---
class Deliverer:
    def __init__(self, gh: GitHubClient, db: Database, comment_delay: float = 1.0,
                 close_previous: bool = True, sleep: Callable[[float], None] = time.sleep):
        self.gh = gh
        self.db = db
        self.comment_delay = comment_delay  # GitHub asks for ~1s between content-creating requests
        self.close_previous = close_previous
        self.sleep = sleep

    def _find_or_create_issue(self, content: DigestContent, result: DeliveryResult) -> tuple[int, bool]:
        """-> (issue number, resumed?)"""
        saved = self.db.get_digest_issue(content.run_id)
        if saved is not None:
            result.issue_url = saved["issue_url"] or ""
            return saved["issue_number"], True

        marker = _RUN_MARKER.format(run_id=content.run_id)
        for issue in self.gh.list_issues(LABEL):  # created on GitHub, but we crashed before saving it?
            if marker in (issue.get("body") or ""):
                log.info("Found run %d's issue #%d on GitHub (not in the DB yet): resuming it",
                         content.run_id, issue["number"])
                self.db.save_digest_issue(content.run_id, issue["number"], issue.get("html_url", ""))
                result.issue_url = issue.get("html_url", "")
                return issue["number"], True

        self.gh.ensure_label(LABEL, description="Weekly research digest")
        issue = self.gh.create_issue(content.title, content.body, [LABEL])
        self.db.save_digest_issue(content.run_id, issue["number"], issue.get("html_url", ""))  # save at once
        result.issue_url = issue.get("html_url", "")
        log.info("Opened issue #%d: %s", issue["number"], content.title)
        return issue["number"], False

    def _recover_from_github(self, run_id: int, number: int, done: dict[str, int], result: DeliveryResult) -> None:
        """Record comments that GitHub has (by marker) but the DB missed because of a crash."""
        for c in self.gh.list_comments(number):
            m = _PAPER_MARKER_RE.search(c.get("body") or "")
            if m and int(m.group(1)) == run_id and m.group(2) not in done:
                self.db.save_delivery(run_id, m.group(2), number, c["id"])
                done[m.group(2)] = c["id"]
                result.recovered += 1
                log.info("Recovered comment for %s from GitHub (posted before a crash)", m.group(2))

    def deliver(self, content: DigestContent) -> DeliveryResult:
        result = DeliveryResult(content.run_id)
        number, resumed = self._find_or_create_issue(content, result)
        result.issue_number = number

        done = self.db.delivered(content.run_id)
        result.skipped = len(done)
        if resumed and len(done) < len(content.comments):
            self._recover_from_github(content.run_id, number, done, result)

        for paper_id, body in content.comments:
            if paper_id in done:
                continue
            if result.posted:
                self.sleep(self.comment_delay)
            comment = self.gh.create_comment(number, body)
            self.db.save_delivery(content.run_id, paper_id, number, comment["id"])  # save at once
            done[paper_id] = comment["id"]
            result.posted += 1
        self.db.mark_digest_complete(content.run_id)

        if self.close_previous:
            for prev in self.db.open_digests_before(content.run_id):
                try:
                    self.gh.close_issue(prev["issue_number"])
                except GitHubNotFound:
                    pass  # deleted on GitHub: nothing to close
                self.db.mark_digest_closed(prev["run_id"])
                result.closed.append(prev["issue_number"])
        return result


def deliver_run(db: Database, cfg: Config, run_id: int, gh: GitHubClient | None,
                sleep: Callable[[float], None] = time.sleep) -> DeliveryResult:
    """Deliver (or resume) one run. Never raises for GitHub problems: the error is in the result."""
    result = DeliveryResult(run_id)
    if gh is None:
        result.error = "no GitHub client"
        return result
    content = build_content(db, cfg, run_id)
    if not content.comments:
        result.error = "no selected papers"
        return result
    try:
        gh.check_access()
        result = Deliverer(gh, db, cfg.deliver.comment_delay_seconds, cfg.deliver.close_previous,
                           sleep=sleep).deliver(content)
    except GitHubAuthError as e:
        log.error("GitHub delivery failed: %s", e)
        result.error = str(e)
    except GitHubError as e:
        log.error("GitHub delivery stopped (%s). Resume with: python -m digest.deliver --run %d", e, run_id)
        result.error = str(e)
    return result


def log_result(result: DeliveryResult) -> None:
    if result.error:
        return
    extra = f", recovered {result.recovered}" if result.recovered else ""
    extra += f", closed previous #{', #'.join(map(str, result.closed))}" if result.closed else ""
    log.info("Delivered run %d to issue #%s: %d comments posted, %d already there%s. %s",
             result.run_id, result.issue_number, result.posted, result.skipped, extra, result.issue_url)


def main(argv: list[str] | None = None) -> int:
    from dotenv import load_dotenv

    from digest.config import load_config
    from digest.github import make_github

    ap = argparse.ArgumentParser(description="Deliver (or resume) a run's digest as a GitHub issue")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--run", type=int, help="run id (default: newest run not fully delivered)")
    ap.add_argument("--dry-run", action="store_true", help="print the issue, don't post")
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

    with Database(cfg.db_path, in_memory_copy=args.dry_run) as db:
        run_id = args.run or db.latest_undelivered_run()
        if run_id is None:
            log.info("Nothing to deliver: every run with selected papers is already delivered")
            return 0
        if args.dry_run:
            print(preview(build_content(db, cfg, run_id)))
            return 0
        gh = make_github()
        if gh is None:
            return 1
        try:
            result = deliver_run(db, cfg, run_id, gh)
        finally:
            gh.close()
        log_result(result)
        return 1 if result.error else 0


if __name__ == "__main__":
    sys.exit(main())
