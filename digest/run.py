"""Pipeline: fetch -> dedup -> hybrid ranking (top K) -> LLM rerank (top N) -> summaries -> digest
-> GitHub issue.

Usage (from the project folder):
    python -m digest.run                      # real run: arXiv + ranking + rerank + summaries
    python -m digest.run --no-embeddings      # BM25 only for the hybrid stage
    python -m digest.run --no-rerank          # skip the LLM rerank, keep the hybrid top N
    python -m digest.run --no-summarize       # skip PDFs and summaries (digest shows abstracts)
    python -m digest.run --no-deliver         # don't post the digest to GitHub
    python -m digest.run --sample tests/fixtures/sample_papers.json --dry-run   # offline test
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from digest.config import Config, load_config
from digest.db import Database
from digest.deliver import build_content, deliver_run, load_entries, log_result, preview
from digest.fetch import fetch_recent, load_sample
from digest.github import make_github
from digest.llm import make_llm
from digest.pdf import PdfFetcher
from digest.render import render_digest
from digest.rerank import RerankedPaper, RerankResult, hybrid_only, rerank
from digest.retrieval import BM25Retriever, HybridRanker, RankedPaper
from digest.summarize import PaperSummary, Summarizer

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


class StageTimer:
    """Wall-clock seconds per stage, logged at the end: shows where a run spends its time."""

    def __init__(self):
        self.times: dict[str, float] = {}

    @contextmanager
    def __call__(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.times[name] = self.times.get(name, 0.0) + time.perf_counter() - t0

    def line(self) -> str:
        parts = [f"{k} {v:.1f}s" for k, v in self.times.items()]
        return " | ".join(parts + [f"total {sum(self.times.values()):.1f}s"])


def run_rerank(cfg: Config, ranked: list[RankedPaper], llm, enabled: bool) -> RerankResult:
    rc = cfg.rerank
    if not ranked:
        return hybrid_only(ranked, rc.top_n, note="rerank: no candidates")
    if not enabled:
        log.info("LLM rerank off: keeping the hybrid top %d", rc.top_n)
        return hybrid_only(ranked, rc.top_n, note="rerank off")
    if llm is None:  # make_llm already logged why
        return hybrid_only(ranked, rc.top_n, note=f"rerank skipped: no {rc.provider} client")
    log.info("LLM rerank: %s/%s, %d candidates in batches of %d", llm.name, llm.model, len(ranked), rc.batch_size)
    return rerank(ranked, llm, topic=topic_text(cfg), top_n=rc.top_n, batch_size=rc.batch_size)


def topic_text(cfg: Config) -> str:
    return f"{cfg.topic.name}. {cfg.topic.description}"


def run_summarize(cfg: Config, db: Database, selected: list[RerankedPaper], llm) -> list[PaperSummary]:
    fetcher = PdfFetcher(delay_seconds=cfg.summarize.pdf_delay_seconds)
    try:
        summarizer = Summarizer(
            llm, db, fetcher, topic_text(cfg),
            model_key=f"{cfg.rerank.provider}/{cfg.rerank.model}",
            max_input_tokens=cfg.summarize.max_input_tokens,
        )
        summaries = summarizer.summarize_all([it.ranked.paper for it in selected])
    finally:
        fetcher.close()
    log.info("PDF downloads this run: %d, summary cache hits: %d", fetcher.downloads, summarizer.cache_hits)
    return summaries


def summaries_note(summaries: list[PaperSummary]) -> str:
    done = [s for s in summaries if s.summary is not None]
    n_pdf = sum(s.source == "pdf" for s in done)
    n_cached = sum(s.cached for s in done)
    return (f"summaries: {len(done)}/{len(summaries)} ({n_pdf} from PDF, {len(done) - n_pdf} abstract-only, "
            f"{n_cached} cached)")


def print_summaries(summaries: list[PaperSummary]) -> None:
    if not summaries:
        return
    off = all(s.error == "summaries off" for s in summaries)
    print(f"\nDigest ({'summaries off: abstracts only' if off else summaries_note(summaries)})")
    print("-" * 100)
    for i, s in enumerate(summaries, 1):
        if s.summary is None:
            tag = f"no summary: {s.error}" if s.error else "no summary"
        else:
            tag = s.source if not s.fallback_reason else f"abstract: {s.fallback_reason}"
            if s.cached:
                tag += ", cached"
        print(f"{i}. [{tag}] {_short(s.paper.title, 75)}")
        if s.summary is not None:
            print(f"   {_short(s.summary.key_result, 95)}")


def save_digest(text: str, out_dir: Path, run_id: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{datetime.now(timezone.utc):%Y-%m-%d}_run{run_id}.md"
    path.write_text(text, encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Research Digest Agent pipeline")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--sample", help="load papers from a JSON file instead of arXiv (offline)")
    ap.add_argument("--no-embeddings", action="store_true", help="BM25 only, no Gemini calls")
    ap.add_argument("--no-rerank", action="store_true", help="skip LLM reranking, keep the hybrid top N")
    ap.add_argument("--no-summarize", action="store_true", help="skip PDF download and summaries")
    ap.add_argument("--no-deliver", action="store_true", help="don't post the digest to GitHub")
    ap.add_argument("--include-seen", action="store_true", help="rank papers even if seen in an earlier run")
    ap.add_argument("--dry-run", action="store_true",
                    help="write nothing: no DB changes, no output files; print the GitHub issue instead of posting")
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
    timer = StageTimer()
    rerank_on = cfg.rerank.enabled and not args.no_rerank
    summarize_on = cfg.summarize.enabled and not args.no_summarize

    # --dry-run works on an in-memory snapshot: same reads as a real run, nothing written to disk
    with Database(cfg.db_path, in_memory_copy=args.dry_run) as db:
        # 1. fetch
        with timer("fetch"):
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
        with timer("rank"):
            ranker, embedder = build_ranker(cfg, db, use_embeddings=not args.no_embeddings)
            log.info("Retrievers: %s", ", ".join(ranker.names))
            ranked = ranker.rank(cfg.topic.ranking_query(), new, cfg.ranking.top_k)
        if embedder is not None:
            log.info("Gemini embedding API calls this run: %d", embedder.api_calls)
        print_ranking(ranked)

        # one LLM client for rerank + summarize: one key, one call counter
        llm = make_llm(cfg.rerank.provider, cfg.rerank.model) if ranked and (rerank_on or summarize_on) else None

        # 4. LLM rerank (falls back to hybrid order on any failure)
        with timer("rerank"):
            result = run_rerank(cfg, ranked, llm, enabled=rerank_on)
        rerank_calls = llm.calls if llm is not None else 0
        log.info(result.note)

        # 5. summarize the selected papers (falls back to the abstract on any failure)
        summaries: list[PaperSummary] = []
        if summarize_on and result.top:
            with timer("summarize"):
                summaries = run_summarize(cfg, db, result.top, llm)
            log.info(summaries_note(summaries))
        elif result.top:
            log.info("Summaries off: the digest shows abstracts")
            summaries = [PaperSummary(it.ranked.paper, None, "abstract", error="summaries off") for it in result.top]
        if llm is not None:
            log.info("LLM calls this run: %d (rerank %d, summarize %d)",
                     llm.calls, rerank_calls, llm.calls - rerank_calls)

        # 6. record the run. On --dry-run this goes to the throwaway in-memory DB, so the digest
        #    and the GitHub preview below are built by exactly the code a real run uses.
        notes = result.note + (f"; {summaries_note(summaries)}" if summarize_on and summaries else "")
        db.save_papers(new)
        run_id = db.log_run(len(papers), len(new), len(ranked), ranker.active, notes=notes)
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

        # 7. files (saved before printing, so a console problem can never lose a paid-for run)
        if args.dry_run:
            log.info("Dry run: nothing written (no database changes, no candidates or digest file)")
        else:
            log.info("Run #%d logged. Papers in DB: %d", run_id, db.count_papers())
            if result.items:
                out = save_candidates(result, cfg.db_path.parent / "candidates", run_id)
                log.info("Saved candidates to %s", out)
            if result.top:
                ranked_by = f"{llm.name}/{llm.model}" if result.used_llm and llm is not None else "hybrid score"
                text = render_digest(load_entries(db, cfg, run_id), cfg.topic.name, run_id,
                                     datetime.now(timezone.utc).date(), ranked_by)
                out = save_digest(text, cfg.db_path.parent / "digests", run_id)
                log.info("Saved digest to %s", out)

        # 8. deliver to GitHub (a failure is logged with a resume command; the digest file is already saved)
        deliver_on = cfg.deliver.provider == "github" and not args.no_deliver and bool(result.top)
        delivery_preview = ""
        if deliver_on and args.dry_run:
            delivery_preview = preview(build_content(db, cfg, run_id))
        elif deliver_on:
            with timer("deliver"):
                gh = make_github()
                if gh is not None:
                    try:
                        delivery = deliver_run(db, cfg, run_id, gh)
                    finally:
                        gh.close()
                    log_result(delivery)
        elif result.top:
            log.info("Delivery off (%s)", "--no-deliver" if args.no_deliver else f"provider: {cfg.deliver.provider}")

        log.info("Timing: %s", timer.line())
        if delivery_preview:
            print("\nGitHub issue preview (dry run: nothing posted)")
            print(delivery_preview)
        print_rerank(result)
        print_summaries(summaries)

    return 0


if __name__ == "__main__":
    sys.exit(main())
