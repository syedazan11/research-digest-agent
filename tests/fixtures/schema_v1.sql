-- digest.db schema 1 (single topic, Days 4-6), copied from main for migration tests
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

-- 👍/👎 per paper, filled by `python -m digest.feedback` from GitHub reactions (Day 4+).
-- Older DBs lack the last three columns; Database._migrate() adds them.
CREATE TABLE IF NOT EXISTS feedback (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id    TEXT NOT NULL REFERENCES papers(id),
    rating      INTEGER NOT NULL CHECK (rating IN (-1, 1)),
    created_at  TEXT NOT NULL,          -- when the reaction was made
    comment_id  INTEGER,                -- the GitHub comment the reaction is on
    source      TEXT,                   -- 'github'
    updated_at  TEXT
);

-- One GitHub issue per run. Saved the moment the issue is created, so a crash can't open a second one.
CREATE TABLE IF NOT EXISTS digest_issues (
    run_id        INTEGER PRIMARY KEY REFERENCES runs(id),
    issue_number  INTEGER NOT NULL,
    issue_url     TEXT,
    created_at    TEXT NOT NULL,
    completed_at  TEXT,                 -- every paper's comment posted
    closed_at     TEXT                  -- closed when the next digest went out
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

-- Per-run ranking snapshot: hybrid vs LLM, and what was selected. Ground truth for Precision@5 evals.
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

-- Summaries are cached so a paper is never summarized twice by the same model.
-- An abstract-only summary is final: it isn't retried from the PDF on later runs.
CREATE TABLE IF NOT EXISTS summaries (
    paper_id         TEXT NOT NULL,
    model            TEXT NOT NULL,     -- "provider/model", e.g. "groq/openai/gpt-oss-120b"
    summary          TEXT NOT NULL,     -- JSON: problem, method, key_result, why_it_matters, limitations
    source           TEXT NOT NULL CHECK (source IN ('pdf', 'abstract')),
    fallback_reason  TEXT,              -- download_failed | not_pdf | no_text | no_sections; NULL for pdf
    created_at       TEXT NOT NULL,
    PRIMARY KEY (paper_id, model)
);
CREATE UNIQUE INDEX IF NOT EXISTS feedback_github_paper ON feedback(paper_id) WHERE source = 'github';
