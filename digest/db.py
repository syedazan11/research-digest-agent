"""SQLite memory: papers, per-topic "seen" sets, cached embeddings and summaries, runs, deliveries, feedback.

Schema versions (PRAGMA user_version):
  1  single topic (Days 1-6). Databases from then report user_version 0; detected by shape.
  2  multiple topics: `topic_papers` is the per-topic "seen" set; runs, digest_issues and feedback
     carry a topic; summaries are cached per (paper, model, topic) because why_it_matters depends
     on the topic. rankings and deliveries get their topic through run_id (one run = one topic).

Migrating 1 -> 2 moves every existing row to one topic (the config's first topic), inside a single
transaction, after writing a .bak copy next to the file.
"""

from __future__ import annotations

import json
import logging
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

import numpy as np

from digest.models import Paper

if TYPE_CHECKING:
    from digest.config import Config

log = logging.getLogger(__name__)

SCHEMA_VERSION = 2

SCHEMA = """
-- Paper metadata, shared by all topics. Whether a topic has "seen" a paper is in topic_papers.
CREATE TABLE IF NOT EXISTS papers (
    id              TEXT PRIMARY KEY,
    title           TEXT NOT NULL,
    abstract        TEXT NOT NULL,
    authors         TEXT NOT NULL,      -- JSON list
    published       TEXT NOT NULL,      -- ISO datetime
    categories      TEXT NOT NULL,      -- JSON list
    pdf_url         TEXT,
    url             TEXT,
    first_seen      TEXT NOT NULL,      -- first time any topic saw it
    sent_in_digest  TEXT                -- legacy (schema 1); per-topic value lives in topic_papers
);

-- Per-topic dedup: a paper can be new for one topic and already seen by another.
CREATE TABLE IF NOT EXISTS topic_papers (
    topic           TEXT NOT NULL,
    paper_id        TEXT NOT NULL REFERENCES papers(id),
    first_seen      TEXT NOT NULL,
    sent_in_digest  TEXT,               -- date it was posted in this topic's digest, NULL if never
    PRIMARY KEY (topic, paper_id)
);

-- Embeddings are cached so we never pay to embed the same text twice. Topic-independent.
CREATE TABLE IF NOT EXISTS embeddings (
    key     TEXT NOT NULL,              -- paper id, or "query:<hash>"
    model   TEXT NOT NULL,
    dims    INTEGER NOT NULL,
    vector  BLOB NOT NULL,              -- float32 bytes
    PRIMARY KEY (key, model, dims)
);

-- 👍/👎 per (topic, paper), filled by `python -m digest.feedback` from GitHub reactions.
CREATE TABLE IF NOT EXISTS feedback (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id    TEXT NOT NULL REFERENCES papers(id),
    rating      INTEGER NOT NULL CHECK (rating IN (-1, 1)),
    created_at  TEXT NOT NULL,          -- when the reaction was made
    comment_id  INTEGER,                -- the GitHub comment the reaction is on
    source      TEXT,                   -- 'github'
    updated_at  TEXT,
    topic       TEXT
);

-- One GitHub issue per run (= per topic per week). Saved the moment it's created.
CREATE TABLE IF NOT EXISTS digest_issues (
    run_id        INTEGER PRIMARY KEY REFERENCES runs(id),
    issue_number  INTEGER NOT NULL,
    issue_url     TEXT,
    created_at    TEXT NOT NULL,
    completed_at  TEXT,                 -- every paper's comment posted
    closed_at     TEXT,                 -- closed when this topic's next digest went out
    topic         TEXT
);

-- One row per posted comment, saved as each one is posted: a re-run resumes, never double-posts.
CREATE TABLE IF NOT EXISTS deliveries (
    run_id        INTEGER NOT NULL REFERENCES runs(id),
    paper_id      TEXT NOT NULL,
    issue_number  INTEGER NOT NULL,
    comment_id    INTEGER NOT NULL,
    posted_at     TEXT NOT NULL,
    PRIMARY KEY (run_id, paper_id)
);

-- One row per topic per pipeline run: your proof it ran unattended.
CREATE TABLE IF NOT EXISTS runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at  TEXT NOT NULL,
    n_fetched   INTEGER NOT NULL,
    n_new       INTEGER NOT NULL,
    n_ranked    INTEGER NOT NULL,
    retrievers  TEXT NOT NULL,          -- JSON list of retriever names used
    notes       TEXT,
    topic       TEXT
);

-- Per-run ranking snapshot: hybrid vs LLM, and what was selected.
CREATE TABLE IF NOT EXISTS rankings (
    run_id        INTEGER NOT NULL REFERENCES runs(id),
    paper_id      TEXT NOT NULL,
    hybrid_rank   INTEGER NOT NULL,
    hybrid_score  REAL NOT NULL,
    llm_score     REAL,                 -- NULL if the LLM didn't score it (or rerank was off)
    reason        TEXT,
    final_rank    INTEGER,              -- 1..top_n if selected, else NULL
    selected      INTEGER NOT NULL CHECK (selected IN (0, 1)),
    PRIMARY KEY (run_id, paper_id)
);

-- Summaries are cached per (paper, model, topic): why_it_matters is written relative to the topic.
-- An abstract-only summary is final: it isn't retried from the PDF on later runs.
CREATE TABLE IF NOT EXISTS summaries (
    paper_id         TEXT NOT NULL,
    model            TEXT NOT NULL,     -- "provider/model", e.g. "groq/openai/gpt-oss-120b"
    topic            TEXT NOT NULL,
    summary          TEXT NOT NULL,     -- JSON: problem, method, key_result, why_it_matters, limitations
    source           TEXT NOT NULL CHECK (source IN ('pdf', 'abstract')),
    fallback_reason  TEXT,              -- download_failed | not_pdf | no_text | no_sections; NULL for pdf
    created_at       TEXT NOT NULL,
    PRIMARY KEY (paper_id, model, topic)
);
"""

# Created after migration: they reference columns old databases only get during it.
INDEXES = """
CREATE UNIQUE INDEX IF NOT EXISTS feedback_github_topic_paper ON feedback(topic, paper_id) WHERE source = 'github';
CREATE INDEX IF NOT EXISTS runs_topic ON runs(topic);
CREATE INDEX IF NOT EXISTS digest_issues_topic ON digest_issues(topic);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class MigrationError(RuntimeError):
    pass


class Database:
    def __init__(self, path: str | Path, in_memory_copy: bool = False, default_topic: str | None = None):
        """default_topic: the topic used when a method isn't told one, and the topic existing (schema 1)
        data is migrated to. Production code passes the config's first topic (see Database.for_config).

        in_memory_copy=True (used by --dry-run): work on a throwaway in-memory snapshot of the DB.
        Reads (dedup, caches) behave exactly like a real run; every write vanishes on close."""
        path = Path(path)
        self.path = path
        self.default_topic = default_topic
        if in_memory_copy:
            self.conn = sqlite3.connect(":memory:")
            if path.exists():
                src = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
                try:
                    src.backup(self.conn)
                finally:
                    src.close()
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self._migrate(on_disk=not in_memory_copy)

    @classmethod
    def for_config(cls, cfg: "Config", in_memory_copy: bool = False) -> "Database":
        return cls(cfg.db_path, in_memory_copy=in_memory_copy, default_topic=cfg.topic.slug)

    # --- schema ---
    def _cols(self, table: str) -> set[str]:
        return {r["name"] for r in self.conn.execute(f"PRAGMA table_info({table})")}

    def _tables(self) -> set[str]:
        return {r["name"] for r in self.conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}

    def _migrate(self, on_disk: bool) -> None:
        version = self.conn.execute("PRAGMA user_version").fetchone()[0]
        legacy = "runs" in self._tables() and "topic" not in self._cols("runs")
        if version >= SCHEMA_VERSION and not legacy:
            self.conn.executescript(SCHEMA + INDEXES)
            return
        if legacy:
            self._migrate_1_to_2(on_disk)
        self.conn.executescript(SCHEMA + INDEXES)
        self.conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        self.conn.commit()

    def _migrate_1_to_2(self, on_disk: bool) -> None:
        """Single-topic -> multi-topic. All existing rows go to `default_topic`, in one transaction."""
        counts = {t: self.conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                  for t in ("papers", "runs") if t in self._tables()}
        if any(counts.values()) and not self.default_topic:
            raise MigrationError(
                "This database is from before multi-topic support and needs to know which topic its data "
                "belongs to. Open it with Database.for_config(cfg) (topics[0]) or default_topic=<slug>.")
        topic = self.default_topic or "default"
        if on_disk and self.path.exists() and any(counts.values()):
            backup = self.path.with_name(self.path.name + ".bak-schema1")
            if not backup.exists():
                self.conn.commit()
                shutil.copy2(self.path, backup)
                log.info("Backed up %s to %s before migrating", self.path.name, backup.name)

        self.conn.commit()
        self.conn.execute("BEGIN")
        try:
            for table, kind in (("runs", "TEXT"), ("digest_issues", "TEXT"), ("feedback", "TEXT")):
                if table in self._tables() and "topic" not in self._cols(table):
                    self.conn.execute(f"ALTER TABLE {table} ADD COLUMN topic {kind}")
            if "feedback" in self._tables():
                for name, kind in (("comment_id", "INTEGER"), ("source", "TEXT"), ("updated_at", "TEXT")):
                    if name not in self._cols("feedback"):  # schema 0 (pre-Day 4)
                        self.conn.execute(f"ALTER TABLE feedback ADD COLUMN {name} {kind}")
                self.conn.execute("DROP INDEX IF EXISTS feedback_github_paper")  # was unique on paper_id only
                self.conn.execute("UPDATE feedback SET topic = ? WHERE topic IS NULL", (topic,))
            self.conn.execute("UPDATE runs SET topic = ? WHERE topic IS NULL", (topic,))
            if "digest_issues" in self._tables():
                self.conn.execute("UPDATE digest_issues SET topic = ? WHERE topic IS NULL", (topic,))

            self.conn.execute("""CREATE TABLE IF NOT EXISTS topic_papers (
                topic TEXT NOT NULL, paper_id TEXT NOT NULL REFERENCES papers(id), first_seen TEXT NOT NULL,
                sent_in_digest TEXT, PRIMARY KEY (topic, paper_id))""")
            if "papers" in self._tables():
                self.conn.execute(
                    """INSERT OR IGNORE INTO topic_papers (topic, paper_id, first_seen, sent_in_digest)
                       SELECT ?, id, first_seen, sent_in_digest FROM papers""", (topic,))

            if "summaries" in self._tables() and "topic" not in self._cols("summaries"):
                # SQLite can't change a primary key in place: rebuild the table
                self.conn.execute("""CREATE TABLE summaries_v2 (
                    paper_id TEXT NOT NULL, model TEXT NOT NULL, topic TEXT NOT NULL, summary TEXT NOT NULL,
                    source TEXT NOT NULL CHECK (source IN ('pdf', 'abstract')), fallback_reason TEXT,
                    created_at TEXT NOT NULL, PRIMARY KEY (paper_id, model, topic))""")
                self.conn.execute(
                    """INSERT INTO summaries_v2 (paper_id, model, topic, summary, source, fallback_reason, created_at)
                       SELECT paper_id, model, ?, summary, source, fallback_reason, created_at FROM summaries""",
                    (topic,))
                self.conn.execute("DROP TABLE summaries")
                self.conn.execute("ALTER TABLE summaries_v2 RENAME TO summaries")
            self.conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise
        log.info("Migrated database to schema %d: existing data now belongs to topic '%s' (%s)",
                 SCHEMA_VERSION, topic, ", ".join(f"{k}: {v}" for k, v in counts.items()))

    def _topic(self, topic: str | None) -> str:
        t = topic or self.default_topic
        if not t:
            raise ValueError("no topic given and the Database has no default_topic")
        return t

    # --- context manager so `with Database(...) as db:` always closes ---
    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self.conn.close()

    # --- papers / per-topic dedup ---
    def known_ids(self, ids: Iterable[str], topic: str | None = None) -> set[str]:
        """Ids this topic has already seen (another topic seeing a paper doesn't count)."""
        t = self._topic(topic)
        ids = list(ids)
        found: set[str] = set()
        for i in range(0, len(ids), 500):  # stay under SQLite's variable limit
            chunk = ids[i : i + 500]
            marks = ",".join("?" * len(chunk))
            rows = self.conn.execute(
                f"SELECT paper_id FROM topic_papers WHERE topic = ? AND paper_id IN ({marks})", [t, *chunk])
            found.update(r["paper_id"] for r in rows)
        return found

    def filter_new(self, papers: list[Paper], topic: str | None = None) -> list[Paper]:
        """Drop papers this topic already ranked in an earlier run."""
        known = self.known_ids((p.id for p in papers), topic)
        return [p for p in papers if p.id not in known]

    def save_papers(self, papers: Iterable[Paper], topic: str | None = None) -> int:
        """Store paper metadata (shared) and mark the papers as seen by this topic. Returns new-for-topic count."""
        t = self._topic(topic)
        papers = list(papers)
        now = _now()
        self.conn.executemany(
            """INSERT OR IGNORE INTO papers
               (id, title, abstract, authors, published, categories, pdf_url, url, first_seen)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [(p.id, p.title, p.abstract, json.dumps(p.authors), p.published.isoformat(), json.dumps(p.categories),
              p.pdf_url, p.url, now) for p in papers],
        )
        cur = self.conn.executemany(
            "INSERT OR IGNORE INTO topic_papers (topic, paper_id, first_seen) VALUES (?, ?, ?)",
            [(t, p.id, now) for p in papers],
        )
        self.conn.commit()
        return cur.rowcount

    def count_papers(self, topic: str | None = None) -> int:
        """All stored papers, or the ones a given topic has seen."""
        if topic is None:
            return self.conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
        return self.conn.execute("SELECT COUNT(*) FROM topic_papers WHERE topic = ?", (topic,)).fetchone()[0]

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
    def log_run(self, n_fetched: int, n_new: int, n_ranked: int, retrievers: list[str], notes: str = "",
                topic: str | None = None) -> int:
        cur = self.conn.execute(
            """INSERT INTO runs (started_at, n_fetched, n_new, n_ranked, retrievers, notes, topic)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (_now(), n_fetched, n_new, n_ranked, json.dumps(retrievers), notes, self._topic(topic)),
        )
        self.conn.commit()
        return cur.lastrowid

    def run_topic(self, run_id: int) -> str | None:
        row = self.conn.execute("SELECT topic FROM runs WHERE id = ?", (run_id,)).fetchone()
        return None if row is None else row["topic"]

    # --- rankings ---
    def save_rankings(self, run_id: int, rows: Iterable[dict]) -> None:
        """rows: dicts with paper_id, hybrid_rank, hybrid_score, llm_score, reason, final_rank."""
        self.conn.executemany(
            """INSERT OR REPLACE INTO rankings
               (run_id, paper_id, hybrid_rank, hybrid_score, llm_score, reason, final_rank, selected)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (run_id, r["paper_id"], r["hybrid_rank"], r["hybrid_score"], r.get("llm_score"),
                 r.get("reason") or None, r.get("final_rank"), int(r.get("final_rank") is not None))
                for r in rows
            ],
        )
        self.conn.commit()

    def get_rankings(self, run_id: int) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM rankings WHERE run_id = ? ORDER BY hybrid_rank", (run_id,)
        ).fetchall()

    # --- summaries cache (per topic) ---
    def get_summary(self, paper_id: str, model: str, topic: str | None = None) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM summaries WHERE paper_id = ? AND model = ? AND topic = ?",
            (paper_id, model, self._topic(topic)),
        ).fetchone()

    def put_summary(self, paper_id: str, model: str, summary_json: str, source: str,
                    fallback_reason: str | None = None, topic: str | None = None) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO summaries
               (paper_id, model, topic, summary, source, fallback_reason, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (paper_id, model, self._topic(topic), summary_json, source, fallback_reason, _now()),
        )
        self.conn.commit()

    # --- runs / papers lookups (delivery rebuilds a run's digest from the DB) ---
    def get_run(self, run_id: int) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()

    def get_paper(self, paper_id: str) -> Paper | None:
        row = self.conn.execute("SELECT * FROM papers WHERE id = ?", (paper_id,)).fetchone()
        if row is None:
            return None
        return Paper.from_dict({**dict(row), "authors": json.loads(row["authors"]),
                                "categories": json.loads(row["categories"])})

    def selected_rankings(self, run_id: int) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM rankings WHERE run_id = ? AND selected = 1 ORDER BY final_rank", (run_id,)
        ).fetchall()

    def latest_undelivered_run(self, topic: str | None = None) -> int | None:
        """Newest run (of `topic`, or of any topic if None) that selected papers but isn't fully delivered."""
        sql = """SELECT r.run_id FROM rankings r
                 JOIN runs ON runs.id = r.run_id
                 LEFT JOIN digest_issues d ON d.run_id = r.run_id
                 WHERE r.selected = 1 AND d.completed_at IS NULL"""
        args: tuple = ()
        if topic is not None:
            sql += " AND runs.topic = ?"
            args = (topic,)
        row = self.conn.execute(sql + " ORDER BY r.run_id DESC LIMIT 1", args).fetchone()
        return None if row is None else row["run_id"]

    # --- GitHub delivery ---
    def get_digest_issue(self, run_id: int) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM digest_issues WHERE run_id = ?", (run_id,)).fetchone()

    def save_digest_issue(self, run_id: int, issue_number: int, issue_url: str = "") -> None:
        self.conn.execute(
            """INSERT OR IGNORE INTO digest_issues (run_id, issue_number, issue_url, created_at, topic)
               VALUES (?, ?, ?, ?, (SELECT topic FROM runs WHERE id = ?))""",
            (run_id, issue_number, issue_url, _now(), run_id),
        )
        self.conn.commit()

    def mark_digest_complete(self, run_id: int) -> None:
        self.conn.execute("UPDATE digest_issues SET completed_at = ? WHERE run_id = ? AND completed_at IS NULL",
                          (_now(), run_id))
        self.conn.commit()

    def mark_digest_closed(self, run_id: int) -> None:
        self.conn.execute("UPDATE digest_issues SET closed_at = ? WHERE run_id = ?", (_now(), run_id))
        self.conn.commit()

    def open_digests_before(self, run_id: int) -> list[sqlite3.Row]:
        """Still-open digest issues of the SAME topic as `run_id`, from earlier runs."""
        return self.conn.execute(
            """SELECT * FROM digest_issues
               WHERE run_id < ? AND closed_at IS NULL AND topic = (SELECT topic FROM runs WHERE id = ?)
               ORDER BY run_id""", (run_id, run_id)
        ).fetchall()

    def delivered(self, run_id: int) -> dict[str, int]:
        """paper_id -> comment_id for comments already posted in this run."""
        rows = self.conn.execute("SELECT paper_id, comment_id FROM deliveries WHERE run_id = ?", (run_id,))
        return {r["paper_id"]: r["comment_id"] for r in rows}

    def save_delivery(self, run_id: int, paper_id: str, issue_number: int, comment_id: int) -> None:
        now = _now()
        self.conn.execute(
            """INSERT OR IGNORE INTO deliveries (run_id, paper_id, issue_number, comment_id, posted_at)
               VALUES (?, ?, ?, ?, ?)""",
            (run_id, paper_id, issue_number, comment_id, now),
        )
        self.conn.execute(
            """UPDATE topic_papers SET sent_in_digest = ?
               WHERE paper_id = ? AND sent_in_digest IS NULL AND topic = (SELECT topic FROM runs WHERE id = ?)""",
            (now[:10], paper_id, run_id))
        self.conn.commit()

    def recent_deliveries(self, since_iso: str, topic: str | None = None) -> list[sqlite3.Row]:
        """Posted comments since a date, each with its run's topic."""
        sql = """SELECT d.*, r.topic AS topic FROM deliveries d JOIN runs r ON r.id = d.run_id
                 WHERE d.posted_at >= ?"""
        args: tuple = (since_iso,)
        if topic is not None:
            sql += " AND r.topic = ?"
            args += (topic,)
        return self.conn.execute(sql + " ORDER BY d.posted_at", args).fetchall()

    # --- feedback (per topic) ---
    def upsert_feedback(self, paper_id: str, rating: int, comment_id: int, reacted_at: str,
                        topic: str | None = None) -> bool:
        """Insert or update the GitHub rating for (topic, paper). Returns True if anything changed."""
        t = self._topic(topic)
        old = self.get_feedback(paper_id, t)
        if old is not None and old["rating"] == rating and old["comment_id"] == comment_id:
            return False
        self.conn.execute(
            """INSERT INTO feedback (paper_id, rating, created_at, comment_id, source, updated_at, topic)
               VALUES (?, ?, ?, ?, 'github', ?, ?)
               ON CONFLICT(topic, paper_id) WHERE source = 'github'
               DO UPDATE SET rating = excluded.rating, created_at = excluded.created_at,
                             comment_id = excluded.comment_id, updated_at = excluded.updated_at""",
            (paper_id, rating, reacted_at, comment_id, _now(), t),
        )
        self.conn.commit()
        return True

    def delete_feedback(self, paper_id: str, topic: str | None = None) -> bool:
        cur = self.conn.execute("DELETE FROM feedback WHERE paper_id = ? AND source = 'github' AND topic = ?",
                                (paper_id, self._topic(topic)))
        self.conn.commit()
        return cur.rowcount > 0

    def get_feedback(self, paper_id: str, topic: str | None = None) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM feedback WHERE paper_id = ? AND source = 'github' AND topic = ?",
            (paper_id, self._topic(topic)),
        ).fetchone()

    def feedback_rows(self, topic: str | None = None) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM feedback WHERE source = 'github' AND topic = ?", (self._topic(topic),)
        ).fetchall()
