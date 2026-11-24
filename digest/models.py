"""Data models shared across the pipeline."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import datetime

_VERSION_SUFFIX = re.compile(r"v\d+$")


def strip_version(arxiv_id: str) -> str:
    """'2411.01234v2' -> '2411.01234'. We dedup on the base id so a v2 isn't treated as a new paper."""
    return _VERSION_SUFFIX.sub("", arxiv_id)


@dataclass
class Paper:
    id: str  # arXiv id without version, e.g. "2411.01234"
    title: str
    abstract: str
    authors: list[str]
    published: datetime  # timezone-aware (UTC)
    categories: list[str]
    pdf_url: str
    url: str

    @property
    def text(self) -> str:
        """The text the retrievers score: title + abstract."""
        return f"{self.title}. {self.abstract}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["published"] = self.published.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Paper":
        return cls(
            id=d["id"],
            title=d["title"],
            abstract=d["abstract"],
            authors=list(d["authors"]),
            published=datetime.fromisoformat(d["published"]),
            categories=list(d["categories"]),
            pdf_url=d.get("pdf_url", ""),
            url=d.get("url", ""),
        )
