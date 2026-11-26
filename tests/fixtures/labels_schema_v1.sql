-- labels.db schema 1 (single topic, Day 6), copied from main for migration tests
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

-- Earlier labelling rounds, moved here by --relabel (kept, so rounds can be compared).
CREATE TABLE IF NOT EXISTS labels_archive (
    round        INTEGER NOT NULL,
    paper_id     TEXT NOT NULL,
    relevant     INTEGER NOT NULL,
    labelled_at  TEXT NOT NULL,
    archived_at  TEXT NOT NULL,
    PRIMARY KEY (round, paper_id)
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
