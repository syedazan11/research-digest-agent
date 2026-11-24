"""SQLite memory: seen papers, cached embeddings, feedback and run history."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

from digest.models import Paper

SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    id              TEXT PRIMARY KEY,
    title           TEXT NOT NULL,
    abstract        TEXT NOT NULL,
    authors         TEXT NOT NULL,      -- JSON list
    published       TEXT NOT NULL,      -- ISO datetime
    categories      TEXT NOT NULL,      -- JSON list
    pdf_url         TEXT,
    url             TEXT,
    first_seen      TEXT NOT NULL,
    sent_in_digest  TEXT                -- date it was sent (Day 3+), NULL if never
);

-- Embeddings are cached so we never pay to embed the same text twice.
CREATE TABLE IF NOT EXISTS embeddings (
    key     TEXT NOT NULL,              -- paper id, or "query:<hash>"
    model   TEXT NOT NULL,
    dims    INTEGER NOT NULL,
    vector  BLOB NOT NULL,              -- float32 bytes
    PRIMARY KEY (key, model, dims)
);

-- Filled by the Telegram poller (Day 4+).
CREATE TABLE IF NOT EXISTS feedback (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id    TEXT NOT NULL REFERENCES papers(id),
    rating      INTEGER NOT NULL CHECK (rating IN (-1, 1)),
    created_at  TEXT NOT NULL
);

-- One row per pipeline run: your proof it ran unattended.
CREATE TABLE IF NOT EXISTS runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at  TEXT NOT NULL,
    n_fetched   INTEGER NOT NULL,
    n_new       INTEGER NOT NULL,
    n_ranked    INTEGER NOT NULL,
    retrievers  TEXT NOT NULL,          -- JSON list of retriever names used
    notes       TEXT
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Database:
    def __init__(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    # --- context manager so `with Database(...) as db:` always closes ---
    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self.conn.close()

    # --- papers ---
    def known_ids(self, ids: Iterable[str]) -> set[str]:
        ids = list(ids)
        found: set[str] = set()
        for i in range(0, len(ids), 500):  # stay under SQLite's variable limit
            chunk = ids[i : i + 500]
            marks = ",".join("?" * len(chunk))
            rows = self.conn.execute(f"SELECT id FROM papers WHERE id IN ({marks})", chunk)
            found.update(r["id"] for r in rows)
        return found

    def filter_new(self, papers: list[Paper]) -> list[Paper]:
        """Drop papers we've already ranked in an earlier run."""
        known = self.known_ids(p.id for p in papers)
        return [p for p in papers if p.id not in known]

    def save_papers(self, papers: Iterable[Paper]) -> int:
        now = _now()
        cur = self.conn.executemany(
            """INSERT OR IGNORE INTO papers
               (id, title, abstract, authors, published, categories, pdf_url, url, first_seen)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    p.id,
                    p.title,
                    p.abstract,
                    json.dumps(p.authors),
                    p.published.isoformat(),
                    json.dumps(p.categories),
                    p.pdf_url,
                    p.url,
                    now,
                )
                for p in papers
            ],
        )
        self.conn.commit()
        return cur.rowcount

    def count_papers(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]

    # --- embedding cache ---
    def get_embedding(self, key: str, model: str, dims: int) -> np.ndarray | None:
        row = self.conn.execute(
            "SELECT vector FROM embeddings WHERE key = ? AND model = ? AND dims = ?",
            (key, model, dims),
        ).fetchone()
        return None if row is None else np.frombuffer(row["vector"], dtype=np.float32)

    def put_embedding(self, key: str, model: str, dims: int, vector: np.ndarray) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO embeddings (key, model, dims, vector) VALUES (?, ?, ?, ?)",
            (key, model, dims, np.asarray(vector, dtype=np.float32).tobytes()),
        )
        self.conn.commit()

    # --- runs ---
    def log_run(self, n_fetched: int, n_new: int, n_ranked: int, retrievers: list[str], notes: str = "") -> int:
        cur = self.conn.execute(
            """INSERT INTO runs (started_at, n_fetched, n_new, n_ranked, retrievers, notes)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (_now(), n_fetched, n_new, n_ranked, json.dumps(retrievers), notes),
        )
        self.conn.commit()
        return cur.lastrowid
