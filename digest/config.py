"""Load config.yaml into typed objects so typos fail early, not halfway through a run."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class TopicConfig:
    name: str
    description: str
    keywords: list[str]
    categories: list[str]

    def ranking_query(self) -> str:
        """The text every retriever compares papers against."""
        return f"{self.name}. {self.description} {' '.join(self.keywords)}".strip()


@dataclass
class FetchConfig:
    days_back: int = 7
    max_results: int = 200


@dataclass
class RankingConfig:
    top_k: int = 20
    bm25_weight: float = 0.4
    embedding_weight: float = 0.6


@dataclass
class EmbeddingConfig:
    provider: str = "gemini"  # "gemini" or "none"
    model: str = "gemini-embedding-001"
    dimensions: int = 768


@dataclass
class Config:
    topic: TopicConfig
    fetch: FetchConfig
    ranking: RankingConfig
    embeddings: EmbeddingConfig
    db_path: Path


def load_config(path: str | Path) -> Config:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    t = raw.get("topic") or {}
    topic = TopicConfig(
        name=str(t.get("name", "")).strip(),
        description=" ".join(str(t.get("description", "")).split()),
        keywords=[str(k) for k in t.get("keywords", [])],
        categories=[str(c) for c in t.get("categories", [])],
    )
    if not topic.name:
        raise ValueError("config: topic.name is required")
    if not topic.keywords and not topic.categories:
        raise ValueError("config: give at least one topic keyword or category")

    f = raw.get("fetch") or {}
    r = raw.get("ranking") or {}
    w = r.get("weights") or {}
    e = raw.get("embeddings") or {}
    d = raw.get("database") or {}

    embeddings = EmbeddingConfig(
        provider=str(e.get("provider", "gemini")).lower(),
        model=str(e.get("model", "gemini-embedding-001")),
        dimensions=int(e.get("dimensions", 768)),
    )
    if embeddings.provider not in {"gemini", "none"}:
        raise ValueError(f"config: unknown embeddings.provider '{embeddings.provider}'")

    return Config(
        topic=topic,
        fetch=FetchConfig(
            days_back=int(f.get("days_back", 7)),
            max_results=int(f.get("max_results", 200)),
        ),
        ranking=RankingConfig(
            top_k=int(r.get("top_k", 20)),
            bm25_weight=float(w.get("bm25", 0.4)),
            embedding_weight=float(w.get("embedding", 0.6)),
        ),
        embeddings=embeddings,
        # relative to the config file, so it works no matter where you run from
        db_path=(path.parent / d.get("path", "data/digest.db")).resolve(),
    )
