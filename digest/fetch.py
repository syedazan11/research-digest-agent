"""Stage 1: pull recent papers from arXiv."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import arxiv

from digest.models import Paper, strip_version

log = logging.getLogger(__name__)


def build_query(keywords: list[str], categories: list[str]) -> str:
    """Build an arXiv search query.

    Example: (cat:cs.CL OR cat:cs.IR) AND (abs:"faithfulness" OR ti:"faithfulness")
    Docs: https://info.arxiv.org/help/api/user-manual.html#query_details
    """
    cat_part = " OR ".join(f"cat:{c}" for c in categories)
    kw_terms = []
    for k in keywords:
        k = k.replace('"', "").strip()
        if k:
            kw_terms += [f'abs:"{k}"', f'ti:"{k}"']
    kw_part = " OR ".join(kw_terms)

    if cat_part and kw_part:
        return f"({cat_part}) AND ({kw_part})"
    return cat_part or kw_part


def _to_paper(r: arxiv.Result) -> Paper:
    return Paper(
        id=strip_version(r.get_short_id()),
        title=" ".join(r.title.split()),
        abstract=" ".join(r.summary.split()),
        authors=[a.name for a in (r.authors or [])],
        published=r.published,
        categories=list(r.categories or []),
        pdf_url=r.pdf_url or "",
        url=r.entry_id,
    )


def fetch_recent(
    keywords: list[str],
    categories: list[str],
    days_back: int,
    max_results: int,
    client: arxiv.Client | None = None,
) -> list[Paper]:
    """Fetch papers submitted in the last `days_back` days, newest first."""
    query = build_query(keywords, categories)
    log.info("arXiv query: %s", query)

    # delay_seconds=3 follows arXiv's API etiquette (no more than 1 request every 3s)
    client = client or arxiv.Client(page_size=100, delay_seconds=3.0, num_retries=3)
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )
    cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)

    papers: dict[str, Paper] = {}
    for r in client.results(search):
        if r.published < cutoff:
            break  # results are sorted newest first, so everything after is older
        p = _to_paper(r)
        papers.setdefault(p.id, p)

    log.info("Fetched %d papers from the last %d days", len(papers), days_back)
    return list(papers.values())


def load_sample(path: str | Path) -> list[Paper]:
    """Offline mode: load papers from a JSON file (used for tests and demos without internet)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    papers = [Paper.from_dict(d) for d in data]
    log.info("Loaded %d sample papers from %s", len(papers), path)
    return papers
