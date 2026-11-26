"""data/labels.db: hand labels for evaluation and feedback learning, plus the eval's LLM score cache.

Why a separate file from data/digest.db? GitHub Actions owns digest.db and commits it every week.
Labels are made locally; two copies of one binary file changing in two places can't be merged.
So you commit labels.db, and CI only ever opens it read-only.

Why a separate label pool? Papers in digest.db count as "seen", so the weekly run would skip them.
A one-off 30-day fetch for labelling must never stop those papers being delivered.

Everything is per topic (schema 2): the same paper can be in two topics' pools with different labels,
and cached LLM scores depend on the topic because the topic is part of the rerank prompt.
Schema-1 files (single topic) migrate to the config's first topic; a read-only open migrates an
in-memory copy instead, so CI never writes the file.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

from digest.models import Paper

if TYPE_CHECKING:
    from digest.config import Config

log = logging.getLogger(__name__)

SCHEMA_VERSION = 2

SCHEMA = """
-- Papers fetched only for labelling. Never touches digest.db, so delivery/dedup are unaffected.
CREATE TABLE IF NOT EXISTS label_pool (
    topic        TEXT NOT NULL,
    paper_id     TEXT NOT NULL,
    paper        TEXT NOT NULL,        -- Paper as JSON
    shuffle_key  TEXT NOT NULL,        -- fixed pseudo-random order: no ranking hints while labelling
    added_at     TEXT NOT NULL,
    PRIMARY KEY (topic, paper_id)
);

-- Your y/n relevance judgements. Kept apart from 👍/👎 reactions (digest.db feedback table).
CREATE TABLE IF NOT EXISTS labels (
    topic        TEXT NOT NULL,
    paper_id     TEXT NOT NULL,
    relevant     INTEGER NOT NULL CHECK (relevant IN (0, 1)),
    labelled_at  TEXT NOT NULL,
    PRIMARY KEY (topic, paper_id)
);

-- Earlier labelling rounds, moved here by --relabel (kept, so rounds can be compared).
CREATE TABLE IF NOT EXISTS labels_archive (
    topic        TEXT NOT NULL,
    round        INTEGER NOT NULL,
    paper_id     TEXT NOT NULL,
    relevant     INTEGER NOT NULL,
    labelled_at  TEXT NOT NULL,
    archived_at  TEXT NOT NULL,
    PRIMARY KEY (topic, round, paper_id)
);

-- LLM rerank scores for the eval, so re-running it costs nothing. Raw score on the prompt's own scale.
CREATE TABLE IF NOT EXISTS llm_scores (
    topic           TEXT NOT NULL,
    paper_id        TEXT NOT NULL,
    model           TEXT NOT NULL,     -- "provider/model"
    prompt_version  TEXT NOT NULL,
    score           REAL NOT NULL,
    reason          TEXT,
    created_at      TEXT NOT NULL,
    PRIMARY KEY (topic, paper_id, model, prompt_version)
);
"""

# schema 1 -> 2: every table gains a topic column at the front of its primary key
_V1_COLUMNS = {
    "label_pool": "paper_id, paper, shuffle_key, added_at",
    "labels": "paper_id, relevant, labelled_at",
    "labels_archive": "round, paper_id, relevant, labelled_at, archived_at",
    "llm_scores": "paper_id, model, prompt_version, score, reason, created_at",
}

SHUFFLE_SEED = "label-order-v1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def shuffle_key(paper_id: str) -> str:
    return hashlib.sha1(f"{SHUFFLE_SEED}:{paper_id}".encode()).hexdigest()


def _is_v1(conn: sqlite3.Connection) -> bool:
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    return "labels" in tables and "topic" not in {r[1] for r in conn.execute("PRAGMA table_info(labels)")}


def migrate_v1(conn: sqlite3.Connection, topic: str) -> None:
    """Rebuild every table with the topic in its key; all existing rows go to `topic`. One transaction."""
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    conn.commit()
    conn.execute("BEGIN")
    try:
        for table, cols in _V1_COLUMNS.items():
            if table in tables:
                conn.execute(f"ALTER TABLE {table} RENAME TO {table}_v1")
        # executescript() would commit; run the CREATEs one by one so they stay inside this transaction
        for stmt in [s for s in SCHEMA.split(";") if s.strip()]:
            conn.execute(stmt)
        for table, cols in _V1_COLUMNS.items():
            if table in tables:
                conn.execute(f"INSERT INTO {table} (topic, {cols}) SELECT ?, {cols} FROM {table}_v1", (topic,))
                conn.execute(f"DROP TABLE {table}_v1")
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        conn.commit()
    except Exception:
        conn.rollback()
        raise


class LabelStore:
    def __init__(self, path: str | Path, read_only: bool = False, default_topic: str | None = None):
        """read_only=True (CI, production ranking): never creates or changes the file. A missing file
        behaves like an empty store; a schema-1 file is migrated in memory only.
        default_topic: used when a method isn't told a topic, and the topic schema-1 data migrates to."""
        path = Path(path)
        self.read_only = read_only
        self.default_topic = default_topic
        if read_only:
            self.conn = sqlite3.connect(":memory:")
            if path.exists():
                src = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
                try:
                    if not _is_v1(src):  # current schema: read the file directly, no copy needed
                        self.conn.close()
                        self.conn = src
                    else:
                        src.backup(self.conn)
                        src.close()
                except Exception:
                    src.close()
                    raise
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row

        if _is_v1(self.conn):
            if not default_topic:
                raise ValueError("labels.db is from before multi-topic support: open it with "
                                 "LabelStore.for_config(cfg) or default_topic=<slug> so its labels keep their topic")
            if not read_only:
                backup = path.with_name(path.name + ".bak-schema1")
                if not backup.exists():
                    shutil.copy2(path, backup)
            migrate_v1(self.conn, default_topic)
            log.info("labels.db: migrated to schema %d, existing labels now belong to '%s'%s",
                     SCHEMA_VERSION, default_topic, " (in memory only)" if read_only else "")
        elif not (read_only and path.exists()):
            self.conn.executescript(SCHEMA)
            self.conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            self.conn.commit()

    @classmethod
    def for_config(cls, cfg: "Config", read_only: bool = False) -> "LabelStore":
        return cls(cfg.labels_path, read_only=read_only, default_topic=cfg.topic.slug)

    def _t(self, topic: str | None) -> str:
        t = topic or self.default_topic
        if not t:
            raise ValueError("no topic given and the LabelStore has no default_topic")
        return t

    def __enter__(self) -> "LabelStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self.conn.close()

    # --- pool ---
    def add_to_pool(self, papers: Iterable[Paper], topic: str | None = None) -> int:
        t = self._t(topic)
        now = _now()
        cur = self.conn.executemany(
            "INSERT OR IGNORE INTO label_pool (topic, paper_id, paper, shuffle_key, added_at) VALUES (?, ?, ?, ?, ?)",
            [(t, p.id, json.dumps(p.to_dict(), ensure_ascii=False), shuffle_key(p.id), now) for p in papers],
        )
        self.conn.commit()
        return cur.rowcount

    def pool(self, topic: str | None = None) -> list[Paper]:
        rows = self.conn.execute("SELECT paper FROM label_pool WHERE topic = ? ORDER BY shuffle_key",
                                 (self._t(topic),))
        return [Paper.from_dict(json.loads(r["paper"])) for r in rows]

    def unlabelled(self, topic: str | None = None) -> list[Paper]:
        rows = self.conn.execute(
            """SELECT p.paper FROM label_pool p
               LEFT JOIN labels l ON l.topic = p.topic AND l.paper_id = p.paper_id
               WHERE p.topic = ? AND l.paper_id IS NULL ORDER BY p.shuffle_key""", (self._t(topic),))
        return [Paper.from_dict(json.loads(r["paper"])) for r in rows]

    # --- labels ---
    def set_label(self, paper_id: str, relevant: bool, topic: str | None = None) -> None:
        self.conn.execute(
            """INSERT INTO labels (topic, paper_id, relevant, labelled_at) VALUES (?, ?, ?, ?)
               ON CONFLICT(topic, paper_id) DO UPDATE SET relevant = excluded.relevant,
                                                          labelled_at = excluded.labelled_at""",
            (self._t(topic), paper_id, int(relevant), _now()),
        )
        self.conn.commit()

    def remove_label(self, paper_id: str, topic: str | None = None) -> None:
        self.conn.execute("DELETE FROM labels WHERE topic = ? AND paper_id = ?", (self._t(topic), paper_id))
        self.conn.commit()

    def archive_labels(self, topic: str | None = None) -> tuple[int, int]:
        """Move this topic's current labels into labels_archive as a new round. The pool and the LLM score
        cache are untouched. Returns (round number, labels archived)."""
        t = self._t(topic)
        rnd = (self.conn.execute("SELECT MAX(round) FROM labels_archive WHERE topic = ?", (t,)).fetchone()[0]
               or 0) + 1
        with self.conn:  # one transaction: either everything moves or nothing does
            cur = self.conn.execute(
                """INSERT INTO labels_archive (topic, round, paper_id, relevant, labelled_at, archived_at)
                   SELECT topic, ?, paper_id, relevant, labelled_at, ? FROM labels WHERE topic = ?""",
                (rnd, _now(), t))
            self.conn.execute("DELETE FROM labels WHERE topic = ?", (t,))
        return rnd, cur.rowcount

    def latest_round(self, topic: str | None = None) -> int | None:
        return self.conn.execute("SELECT MAX(round) FROM labels_archive WHERE topic = ?",
                                 (self._t(topic),)).fetchone()[0]

    def archived_labels(self, rnd: int, topic: str | None = None) -> dict[str, int]:
        rows = self.conn.execute("SELECT paper_id, relevant FROM labels_archive WHERE topic = ? AND round = ?",
                                 (self._t(topic), rnd))
        return {r["paper_id"]: r["relevant"] for r in rows}

    def labels(self, topic: str | None = None) -> dict[str, int]:
        """paper_id -> 1 (relevant) / 0 (not)."""
        rows = self.conn.execute("SELECT paper_id, relevant FROM labels WHERE topic = ?", (self._t(topic),))
        return {r["paper_id"]: r["relevant"] for r in rows}

    def labelled_papers(self, topic: str | None = None) -> list[tuple[Paper, int]]:
        rows = self.conn.execute(
            """SELECT p.paper, l.relevant FROM labels l
               JOIN label_pool p ON p.topic = l.topic AND p.paper_id = l.paper_id
               WHERE l.topic = ? ORDER BY p.shuffle_key""", (self._t(topic),))
        return [(Paper.from_dict(json.loads(r["paper"])), r["relevant"]) for r in rows]

    def stats(self, topic: str | None = None) -> dict[str, int]:
        t = self._t(topic)
        pool = self.conn.execute("SELECT COUNT(*) FROM label_pool WHERE topic = ?", (t,)).fetchone()[0]
        rel = self.conn.execute("SELECT COUNT(*) FROM labels WHERE topic = ? AND relevant = 1", (t,)).fetchone()[0]
        not_rel = self.conn.execute("SELECT COUNT(*) FROM labels WHERE topic = ? AND relevant = 0",
                                    (t,)).fetchone()[0]
        return {"pool": pool, "relevant": rel, "not_relevant": not_rel, "unlabelled": pool - rel - not_rel}

    # --- eval LLM score cache ---
    def get_llm_scores(self, model: str, prompt_version: str, topic: str | None = None) -> dict[str, tuple[float, str]]:
        rows = self.conn.execute(
            "SELECT paper_id, score, reason FROM llm_scores WHERE topic = ? AND model = ? AND prompt_version = ?",
            (self._t(topic), model, prompt_version),
        )
        return {r["paper_id"]: (r["score"], r["reason"] or "") for r in rows}

    def put_llm_score(self, paper_id: str, model: str, prompt_version: str, score: float, reason: str,
                      topic: str | None = None) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO llm_scores (topic, paper_id, model, prompt_version, score, reason, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (self._t(topic), paper_id, model, prompt_version, score, reason, _now()),
        )
        self.conn.commit()
