"""Render the weekly digest as Markdown: data/digests/<date>_run<id>.md.

Markdown is the one format every later stage can reuse: Telegram (Day 4) and GitHub Pages (Day 7).
Everything that came from a paper or the LLM is escaped, so a title with * or _ can't break the
formatting, and injected text can't turn into a live link or image.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Sequence

from digest.rerank import RerankedPaper
from digest.summarize import PaperSummary

MAX_AUTHORS = 5
_MD_SPECIAL = re.compile(r"([\\`*_\[\]<>|#!])")

LABELS = {
    "problem": "Problem",
    "method": "Method",
    "key_result": "Key result",
    "why_it_matters": "Why it matters",
    "limitations": "Limitations",
}


def md(text: str) -> str:
    """Escape Markdown special characters in untrusted text and flatten it to one line."""
    return _MD_SPECIAL.sub(r"\\\1", " ".join(text.split()))


def _authors(names: list[str]) -> str:
    shown = ", ".join(md(n) for n in names[:MAX_AUTHORS])
    return shown + (" et al." if len(names) > MAX_AUTHORS else "")


def _abs_url(url: str) -> str:
    """http://arxiv.org/abs/2611.26077v1 -> https://arxiv.org/abs/2611.26077v1"""
    return url.replace("http://", "https://", 1) if url.startswith("http://arxiv.org/") else url


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
        p = item.ranked.paper
        lines += ["---", "", f"## {i}. {md(p.title)}", ""]
        meta = [f"**Authors:** {_authors(p.authors)}"]
        if p.url:
            meta.append(f"**Link:** {_abs_url(p.url)}")
        if item.llm_score is not None:
            score = f"{item.llm_score:g}/10"
            meta.append(f"**Relevance:** {score} - {md(item.reason)}" if item.reason else f"**Relevance:** {score}")
        lines += ["  \n".join(meta), ""]  # two trailing spaces = Markdown line break

        if ps.summary is None:
            lines += ["*Summary unavailable. Abstract:*", "", f"> {md(p.abstract)}", ""]
            continue
        for field, label in LABELS.items():
            lines.append(f"- **{label}:** {md(getattr(ps.summary, field))}")
        lines.append("")
        if ps.source == "abstract":
            reason = f" ({ps.fallback_reason})" if ps.fallback_reason else ""
            lines += [f"*Summary based on the abstract only{reason}.*", ""]

    if not entries:
        lines.append("No new papers this week.")
    return "\n".join(lines).rstrip() + "\n"
