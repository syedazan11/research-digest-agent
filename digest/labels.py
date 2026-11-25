"""data/labels.db: hand labels for evaluation and feedback learning, plus the eval's LLM score cache.

Why a separate file from data/digest.db? GitHub Actions owns digest.db and commits it every week.
Labels are made locally; two copies of one binary file changing in two places can't be merged.
So you commit labels.db, and CI only ever opens it read-only.

Why a separate label pool? Papers in digest.db's `papers` table count as "seen", so the weekly run
would skip them. A one-off 30-day fetch for labelling must never stop those papers being delivered.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from digest.models import Paper

SCHEMA = """
-- Papers fetched only for labelling. Never touches digest.db, so delivery/dedup are unaffected.
CREATE TABLE IF NOT EXISTS label_pool (
    paper_id     TEXT PRIMARY KEY,
    paper        TEXT NOT NULL,        -- Paper as JSON
    shuffle_key  TEXT NOT NULL,        -- fixed pseudo-random order: no ranking hints while labelling
    added_at     TEXT NOT NULL
);

-- Your y/n relevance judgements. Kept apart from 👍/👎 reactions (digest.db feedback table).
CREATE TABLE IF NOT EXISTS labels (
    paper_id     TEXT PRIMARY KEY,
    relevant     INTEGER NOT NULL CHECK (relevant IN (0, 1)),
    labelled_at  TEXT NOT NULL
);

-- LLM rerank scores for the eval, so re-running it costs nothing. Raw score on the prompt's own scale.
CREATE TABLE IF NOT EXISTS llm_scores (
    paper_id        TEXT NOT NULL,
    model           TEXT NOT NULL,     -- "provider/model"
    prompt_version  TEXT NOT NULL,
    score           REAL NOT NULL,
    reason          TEXT,
    created_at      TEXT NOT NULL,
    PRIMARY KEY (paper_id, model, prompt_version)
);
"""

SHUFFLE_SEED = "label-order-v1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def shuffle_key(paper_id: str) -> str:
    return hashlib.sha1(f"{SHUFFLE_SEED}:{paper_id}".encode()).hexdigest()


class LabelStore:
    def __init__(self, path: str | Path, read_only: bool = False):
        """read_only=True (CI, production ranking): never creates or changes the file. A missing file
        behaves like an empty store."""
        path = Path(path)
        self.read_only = read_only
        if read_only:
            if path.exists():
                self.conn = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
            else:
                self.conn = sqlite3.connect(":memory:")
                self.conn.executescript(SCHEMA)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(path)
            self.conn.executescript(SCHEMA)
        self.conn.row_factory = sqlite3.Row

    def __enter__(self) -> "LabelStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self.conn.close()

    # --- pool ---
    def add_to_pool(self, papers: Iterable[Paper]) -> int:
        now = _now()
        cur = self.conn.executemany(
            "INSERT OR IGNORE INTO label_pool (paper_id, paper, shuffle_key, added_at) VALUES (?, ?, ?, ?)",
            [(p.id, json.dumps(p.to_dict(), ensure_ascii=False), shuffle_key(p.id), now) for p in papers],
        )
        self.conn.commit()
        return cur.rowcount

    def pool(self) -> list[Paper]:
        rows = self.conn.execute("SELECT paper FROM label_pool ORDER BY shuffle_key")
        return [Paper.from_dict(json.loads(r["paper"])) for r in rows]

    def unlabelled(self) -> list[Paper]:
        rows = self.conn.execute(
            """SELECT p.paper FROM label_pool p LEFT JOIN labels l ON l.paper_id = p.paper_id
               WHERE l.paper_id IS NULL ORDER BY p.shuffle_key"""
        )
        return [Paper.from_dict(json.loads(r["paper"])) for r in rows]

    # --- labels ---
    def set_label(self, paper_id: str, relevant: bool) -> None:
        self.conn.execute(
            """INSERT INTO labels (paper_id, relevant, labelled_at) VALUES (?, ?, ?)
               ON CONFLICT(paper_id) DO UPDATE SET relevant = excluded.relevant, labelled_at = excluded.labelled_at""",
            (paper_id, int(relevant), _now()),
        )
        self.conn.commit()

    def remove_label(self, paper_id: str) -> None:
        self.conn.execute("DELETE FROM labels WHERE paper_id = ?", (paper_id,))
        self.conn.commit()

    def labels(self) -> dict[str, int]:
        """paper_id -> 1 (relevant) / 0 (not)."""
        return {r["paper_id"]: r["relevant"] for r in self.conn.execute("SELECT paper_id, relevant FROM labels")}

    def labelled_papers(self) -> list[tuple[Paper, int]]:
        rows = self.conn.execute(
            """SELECT p.paper, l.relevant FROM labels l JOIN label_pool p ON p.paper_id = l.paper_id
               ORDER BY p.shuffle_key"""
        )
        return [(Paper.from_dict(json.loads(r["paper"])), r["relevant"]) for r in rows]

    def stats(self) -> dict[str, int]:
        pool = self.conn.execute("SELECT COUNT(*) FROM label_pool").fetchone()[0]
        rel = self.conn.execute("SELECT COUNT(*) FROM labels WHERE relevant = 1").fetchone()[0]
        not_rel = self.conn.execute("SELECT COUNT(*) FROM labels WHERE relevant = 0").fetchone()[0]
        return {"pool": pool, "relevant": rel, "not_relevant": not_rel, "unlabelled": pool - rel - not_rel}

    # --- eval LLM score cache ---
    def get_llm_scores(self, model: str, prompt_version: str) -> dict[str, tuple[float, str]]:
        rows = self.conn.execute(
            "SELECT paper_id, score, reason FROM llm_scores WHERE model = ? AND prompt_version = ?",
            (model, prompt_version),
        )
        return {r["paper_id"]: (r["score"], r["reason"] or "") for r in rows}

    def put_llm_score(self, paper_id: str, model: str, prompt_version: str, score: float, reason: str) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO llm_scores (paper_id, model, prompt_version, score, reason, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (paper_id, model, prompt_version, score, reason, _now()),
        )
        self.conn.commit()
