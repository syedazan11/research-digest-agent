"""Render papers as Markdown: the weekly digest file and the GitHub issue comments.

Markdown is the one format every later stage reuses: GitHub Issues (Day 4) and GitHub Pages (Day 7).
Both the digest file and the issue comments come from `render_paper`, so they can't drift apart.

Everything that came from a paper or the LLM goes through `md()`:
- Markdown special characters are backslash-escaped, so a title with * or _ can't break formatting
  and injected text can't become a live link or image;
- @mentions and issue references (#123, owner/repo#123, GH-123) get an invisible word joiner
  (U+2060) after the @/#. Backslash-escaping does NOT stop GitHub's autolinker; verified with
  GitHub's POST /markdown renderer (see tests/test_render.py).
"""

from __future__ import annotations

import re
from datetime import date
from typing import Sequence

from digest.rerank import RerankedPaper
from digest.summarize import PaperSummary

MAX_AUTHORS = 5
_MD_SPECIAL = re.compile(r"([\\`*_\[\]<>|#!])")
WORD_JOINER = "⁠"
_MENTION = re.compile(r"(?<![\w.])@(?=[A-Za-z0-9])")  # @user, @org/team; not emails (a@b.com)
_ISSUE_REF = re.compile(r"(#|\bGH-)(?=\d)", re.IGNORECASE)  # #123, owner/repo#123, GH-123

LABELS = {
    "problem": "Problem",
    "method": "Method",
    "key_result": "Key result",
    "why_it_matters": "Why it matters",
    "limitations": "Limitations",
}


def defang(text: str) -> str:
    """Stop GitHub from turning @user into a ping or #123 into an issue link (text still reads the same)."""
    text = _MENTION.sub("@" + WORD_JOINER, text)
    return _ISSUE_REF.sub(lambda m: m.group(1) + WORD_JOINER, text)


def md(text: str) -> str:
    """Make untrusted text safe for GitHub Markdown, flattened to one line."""
    return _MD_SPECIAL.sub(r"\\\1", defang(" ".join(text.split())))


def _authors(names: list[str]) -> str:
    shown = ", ".join(md(n) for n in names[:MAX_AUTHORS])
    return shown + (" et al." if len(names) > MAX_AUTHORS else "")


def abs_url(url: str) -> str:
    """http://arxiv.org/abs/2611.26077v1 -> https://arxiv.org/abs/2611.26077v1"""
    return url.replace("http://", "https://", 1) if url.startswith("http://arxiv.org/") else url


def render_paper(rank: int, item: RerankedPaper, ps: PaperSummary) -> list[str]:
    """One paper's Markdown: heading, authors/link/relevance, five fields (or the abstract)."""
    p = item.ranked.paper
    lines = [f"## {rank}. {md(p.title)}", ""]
    meta = [f"**Authors:** {_authors(p.authors)}"]
    if p.url:
        meta.append(f"**Link:** {abs_url(p.url)}")
    if item.llm_score is not None:
        score = f"{item.llm_score:g}/10"
        meta.append(f"**Relevance:** {score} - {md(item.reason)}" if item.reason else f"**Relevance:** {score}")
    lines += ["  \n".join(meta), ""]  # two trailing spaces = Markdown line break

    if ps.summary is None:
        return lines + ["*Summary unavailable. Abstract:*", "", f"> {md(p.abstract)}", ""]
    for field, label in LABELS.items():
        lines.append(f"- **{label}:** {md(getattr(ps.summary, field))}")
    lines.append("")
    if ps.source == "abstract":
        reason = f" ({ps.fallback_reason})" if ps.fallback_reason else ""
        lines += [f"*Summary based on the abstract only{reason}.*", ""]
    return lines


def render_digest(
    entries: Sequence[tuple[RerankedPaper, PaperSummary]],
    topic: str,
    run_id: int,
    run_date: date,
    ranked_by: str,
) -> str:
    lines = [
        f"# Research digest: {md(topic)}",
        "",
        f"*{run_date:%Y-%m-%d} · run {run_id} · {len(entries)} papers · ranked by {md(ranked_by)}*",
        "",
    ]
    for i, (item, ps) in enumerate(entries, 1):
        lines += ["---", ""] + render_paper(i, item, ps)
    if not entries:
        lines.append("No new papers this week.")
    return "\n".join(lines).rstrip() + "\n"
