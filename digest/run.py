"""Day 1 pipeline: fetch -> dedup -> hybrid ranking -> top-K candidates.

Usage (from the project folder):
    python -m digest.run                      # real run: arXiv + BM25 + Gemini embeddings
    python -m digest.run --no-embeddings      # BM25 only, no API key needed
    python -m digest.run --sample tests/fixtures/sample_papers.json --dry-run   # offline test
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from digest.config import Config, load_config
from digest.db import Database
from digest.fetch import fetch_recent, load_sample
from digest.retrieval import BM25Retriever, HybridRanker, RankedPaper

log = logging.getLogger("digest")


def build_ranker(cfg: Config, db: Database, use_embeddings: bool) -> tuple[HybridRanker, object | None]:
    weighted = [(BM25Retriever(), cfg.ranking.bm25_weight)]
    embedder = None

    if use_embeddings and cfg.embeddings.provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key or api_key == "your-key-here":
            log.warning("GEMINI_API_KEY not set: falling back to BM25 only (see .env.example)")
        else:
            from digest.retrieval.gemini_embed import GeminiEmbeddingRetriever

            embedder = GeminiEmbeddingRetriever(
                db, api_key=api_key, model=cfg.embeddings.model, dimensions=cfg.embeddings.dimensions
            )
            weighted.append((embedder, cfg.ranking.embedding_weight))

    return HybridRanker(weighted), embedder


def print_ranking(ranked: list[RankedPaper]) -> None:
    if not ranked:
        print("\nNo new papers to rank.")
        return
    print(f"\nTop {len(ranked)} candidates")
    print("-" * 100)
    for i, rp in enumerate(ranked, 1):
        comps = "  ".join(f"{k} {v:.2f}" for k, v in rp.components.items())
        title = rp.paper.title if len(rp.paper.title) <= 80 else rp.paper.title[:77] + "..."
        print(f"{i:>2}. [{rp.score:.3f}] {title}")
        print(f"    {rp.paper.published:%Y-%m-%d}  {comps}  {rp.paper.url}")


def save_candidates(ranked: list[RankedPaper], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{datetime.now(timezone.utc):%Y-%m-%d}.json"
    path.write_text(
        json.dumps([rp.to_dict(i) for i, rp in enumerate(ranked, 1)], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Research Digest Agent - Day 1 pipeline")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--sample", help="load papers from a JSON file instead of arXiv (offline)")
    ap.add_argument("--no-embeddings", action="store_true", help="BM25 only, no Gemini calls")
    ap.add_argument("--include-seen", action="store_true", help="rank papers even if seen in an earlier run")
    ap.add_argument("--dry-run", action="store_true", help="don't save papers or log the run to the DB")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("httpx", "urllib3", "google_genai", "arxiv"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    load_dotenv()

    cfg = load_config(args.config)
    log.info("Topic: %s", cfg.topic.name)

    with Database(cfg.db_path) as db:
        # 1. fetch
        if args.sample:
            papers = load_sample(args.sample)
        else:
            try:
                papers = fetch_recent(
                    cfg.topic.keywords, cfg.topic.categories, cfg.fetch.days_back, cfg.fetch.max_results
                )
            except Exception as e:  # network down, arXiv overloaded, etc.
                log.error("Could not reach arXiv (%s: %s). Check your internet, or test offline "
                          "with --sample tests/fixtures/sample_papers.json", type(e).__name__, str(e)[:200])
                return 1

        # 2. dedup
        new = papers if args.include_seen else db.filter_new(papers)
        log.info("%d fetched, %d new (%d already seen)", len(papers), len(new), len(papers) - len(new))

        # 3. rank
        ranker, embedder = build_ranker(cfg, db, use_embeddings=not args.no_embeddings)
        log.info("Retrievers: %s", ", ".join(ranker.names))
        ranked = ranker.rank(cfg.topic.ranking_query(), new, cfg.ranking.top_k)
        if embedder is not None:
            log.info("Gemini embedding API calls this run: %d", embedder.api_calls)

        # 4. output
        print_ranking(ranked)
        if ranked:
            out = save_candidates(ranked, cfg.db_path.parent / "candidates")
            log.info("Saved candidates to %s", out)

        if args.dry_run:
            log.info("Dry run: nothing written to the database")
        else:
            db.save_papers(new)
            run_id = db.log_run(len(papers), len(new), len(ranked), ranker.names)
            log.info("Run #%d logged. Papers in DB: %d", run_id, db.count_papers())

    return 0


if __name__ == "__main__":
    sys.exit(main())
