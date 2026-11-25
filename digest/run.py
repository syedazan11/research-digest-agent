"""Pipeline: fetch -> dedup -> hybrid ranking (top K) -> LLM rerank (top N).

Usage (from the project folder):
    python -m digest.run                      # real run: arXiv + hybrid ranking + LLM rerank
    python -m digest.run --no-embeddings      # BM25 only for the hybrid stage
    python -m digest.run --no-rerank          # skip the LLM, keep the hybrid top N
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
from digest.llm import make_llm
from digest.rerank import RerankResult, hybrid_only, rerank
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


def _short(text: str, n: int) -> str:
    return text if len(text) <= n else text[: n - 3] + "..."


def print_ranking(ranked: list[RankedPaper]) -> None:
    if not ranked:
        print("\nNo new papers to rank.")
        return
    print(f"\nHybrid top {len(ranked)}")
    print("-" * 100)
    for i, rp in enumerate(ranked, 1):
        comps = "  ".join(f"{k} {v:.2f}" for k, v in rp.components.items())
        print(f"{i:>2}. [{rp.score:.3f}] {_short(rp.paper.title, 80)}")
        print(f"    {rp.paper.published:%Y-%m-%d}  {comps}  {rp.paper.url}")


def print_rerank(result: RerankResult) -> None:
    """Before/after table: hybrid rank -> final rank, LLM score, one-line reason. '*' = selected."""
    if not result.items:
        return
    print(f"\nRerank ({result.note})")
    print(" hybrid -> final  llm   title")
    print("-" * 100)
    for it in result.items:
        final = f"{it.final_rank:>2}" if it.selected else " -"
        llm = f"{it.llm_score:>4.1f}" if it.llm_score is not None else "   -"
        mark = "*" if it.selected else " "
        print(f"{mark} #{it.hybrid_rank:<3} ->  {final}   {llm}  {_short(it.ranked.paper.title, 70)}")
        if it.reason:
            print(f"                       {_short(it.reason, 75)}")


def save_candidates(result: RerankResult, out_dir: Path, run_id: int) -> Path:
    """All candidates in final order, with hybrid and LLM fields. Later stages read this file.
    Named by date and run id (2026-11-25_run3.json) so two runs on the same day never overwrite."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{datetime.now(timezone.utc):%Y-%m-%d}_run{run_id}.json"
    path.write_text(
        json.dumps([it.to_dict() for it in result.items], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def run_rerank(cfg: Config, ranked: list[RankedPaper], enabled: bool) -> tuple[RerankResult, object | None]:
    rc = cfg.rerank
    if not ranked:
        return hybrid_only(ranked, rc.top_n, note="rerank: no candidates"), None
    if not enabled:
        log.info("LLM rerank off: keeping the hybrid top %d", rc.top_n)
        return hybrid_only(ranked, rc.top_n, note="rerank off"), None
    llm = make_llm(rc.provider, rc.model)
    if llm is None:  # make_llm already logged why
        return hybrid_only(ranked, rc.top_n, note=f"rerank skipped: no {rc.provider} client"), None
    log.info("LLM rerank: %s/%s, %d candidates in batches of %d", llm.name, llm.model, len(ranked), rc.batch_size)
    result = rerank(ranked, llm, topic=f"{cfg.topic.name}. {cfg.topic.description}",
                    top_n=rc.top_n, batch_size=rc.batch_size)
    return result, llm


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Research Digest Agent pipeline")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--sample", help="load papers from a JSON file instead of arXiv (offline)")
    ap.add_argument("--no-embeddings", action="store_true", help="BM25 only, no Gemini calls")
    ap.add_argument("--no-rerank", action="store_true", help="skip LLM reranking, keep the hybrid top N")
    ap.add_argument("--include-seen", action="store_true", help="rank papers even if seen in an earlier run")
    ap.add_argument("--dry-run", action="store_true", help="don't save papers or log the run to the DB")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Windows consoles default to cp1252, which can't print characters like U+2011 that LLMs
    # (and paper titles) use. Show '?' for those instead of crashing.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    for noisy in ("httpx", "urllib3", "google_genai", "arxiv", "groq"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    load_dotenv()

    cfg = load_config(args.config)
    log.info("Topic: %s", cfg.topic.name)

    # --dry-run works on an in-memory snapshot: same reads as a real run, nothing written to disk
    with Database(cfg.db_path, in_memory_copy=args.dry_run) as db:
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

        # 3. hybrid rank
        ranker, embedder = build_ranker(cfg, db, use_embeddings=not args.no_embeddings)
        log.info("Retrievers: %s", ", ".join(ranker.names))
        ranked = ranker.rank(cfg.topic.ranking_query(), new, cfg.ranking.top_k)
        if embedder is not None:
            log.info("Gemini embedding API calls this run: %d", embedder.api_calls)
        print_ranking(ranked)

        # 4. LLM rerank (falls back to hybrid order on any failure)
        result, llm = run_rerank(cfg, ranked, enabled=cfg.rerank.enabled and not args.no_rerank)
        if llm is not None:
            log.info("LLM API calls this run: %d", llm.calls)
        log.info(result.note)

        # 5. output (saved before printing, so a console problem can never lose a paid-for run)
        if args.dry_run:
            log.info("Dry run: nothing written (no database changes, no candidates file)")
        else:
            db.save_papers(new)
            run_id = db.log_run(len(papers), len(new), len(ranked), ranker.active, notes=result.note)
            db.save_rankings(run_id, (
                {
                    "paper_id": it.ranked.paper.id,
                    "hybrid_rank": it.hybrid_rank,
                    "hybrid_score": it.ranked.score,
                    "llm_score": it.llm_score,
                    "reason": it.reason,
                    "final_rank": it.final_rank,
                }
                for it in result.items
            ))
            log.info("Run #%d logged. Papers in DB: %d", run_id, db.count_papers())
            if result.items:
                out = save_candidates(result, cfg.db_path.parent / "candidates", run_id)
                log.info("Saved candidates to %s", out)

        print_rerank(result)

    return 0


if __name__ == "__main__":
    sys.exit(main())
