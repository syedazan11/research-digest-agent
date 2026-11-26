"""Pipeline, once per topic: fetch -> dedup -> hybrid ranking (top K) -> LLM rerank (top N) -> summaries
-> digest -> GitHub issue.

Usage (from the project folder):
    python -m digest.run                      # every topic in config.yaml
    python -m digest.run --topic ai-agents    # just one topic
    python -m digest.run --no-embeddings      # BM25 only for the hybrid stage
    python -m digest.run --no-rerank          # skip the LLM rerank, keep the hybrid top N
    python -m digest.run --no-summarize       # skip PDFs and summaries (digest shows abstracts)
    python -m digest.run --no-deliver         # don't post the digest to GitHub
    python -m digest.run --strict             # exit 1 if any topic failed or fell back (GitHub Actions)
    python -m digest.run --sample tests/fixtures/sample_papers.json --dry-run   # offline test

Topics are independent: each has its own "seen" papers, run row, digest file and issue. If one topic
crashes, the others still run; --strict turns that into a failed job at the end.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from digest.config import Config, TopicConfig, load_config
from digest.db import Database
from digest.deliver import build_content, deliver_run, load_entries, log_result, preview
from digest.fetch import fetch_recent, load_sample
from digest.github import make_github
from digest.llm import BudgetedLLM, describe_rate_limits, make_llm
from digest.pdf import PdfFetcher
from digest.render import render_digest
from digest.rerank import RerankedPaper, RerankResult, hybrid_only, rerank
from digest.retrieval import BM25Retriever, HybridRanker, RankedPaper
from digest.retrieval.feedback import FeedbackRetriever, load_examples
from digest.summarize import PaperSummary, Summarizer

log = logging.getLogger("digest")


def build_ranker(cfg: Config, db: Database, use_embeddings: bool,
                 topic: TopicConfig | None = None) -> tuple[HybridRanker, object | None]:
    topic = topic or cfg.topic
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

    if cfg.ranking.feedback_weight > 0:
        # each topic learns only from its own reactions and labels
        fb = FeedbackRetriever(load_examples(db, cfg.labels_path, topic.slug))
        c = fb.counts()
        log.info("Feedback retriever: %d liked, %d disliked (%d reactions, %d labels), weight %.2f",
                 c["positive"], c["negative"], c["reaction"], c["label"], cfg.ranking.feedback_weight)
        weighted.append((fb, cfg.ranking.feedback_weight))

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


def _stamp(slug: str, run_id: int) -> str:
    return f"{datetime.now(timezone.utc):%Y-%m-%d}_{slug}_run{run_id}"


def save_candidates(result: RerankResult, out_dir: Path, run_id: int, slug: str = "") -> Path:
    """All candidates in final order, with hybrid and LLM fields. Named by date, topic and run id
    (2026-11-26_ai-agents_run7.json), so runs never overwrite each other."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{_stamp(slug, run_id) if slug else f'{datetime.now(timezone.utc):%Y-%m-%d}_run{run_id}'}.json"
    path.write_text(json.dumps([it.to_dict() for it in result.items], indent=2, ensure_ascii=False),
                    encoding="utf-8")
    return path


class StageTimer:
    """Wall-clock seconds per stage, summed over topics and logged at the end."""

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


def topic_text(topic: Config | TopicConfig) -> str:
    t = topic.topic if isinstance(topic, Config) else topic
    return f"{t.name}. {t.description}"


def run_rerank(cfg: Config, ranked: list[RankedPaper], llm, enabled: bool,
               topic: TopicConfig | None = None) -> RerankResult:
    topic = topic or cfg.topic
    rc = cfg.rerank
    top_n = cfg.top_n(topic)
    if not ranked:
        return hybrid_only(ranked, top_n, note="rerank: no candidates")
    if not enabled:
        log.info("LLM rerank off: keeping the hybrid top %d", top_n)
        return hybrid_only(ranked, top_n, note="rerank off")
    if llm is None:  # make_llm already logged why
        return hybrid_only(ranked, top_n, note=f"rerank skipped: no {rc.provider} client")
    log.info("LLM rerank: %s/%s, %d candidates in batches of %d", llm.name, llm.model, len(ranked), rc.batch_size)
    return rerank(ranked, llm, topic=topic_text(topic), top_n=top_n, batch_size=rc.batch_size,
                  prompt_version=rc.prompt_version)


def run_summarize(cfg: Config, db: Database, selected: list[RerankedPaper], llm,
                  topic: TopicConfig | None = None, fetcher: PdfFetcher | None = None) -> list[PaperSummary]:
    topic = topic or cfg.topic
    own = fetcher is None
    fetcher = fetcher or PdfFetcher(delay_seconds=cfg.summarize.pdf_delay_seconds)
    downloads_before = fetcher.downloads
    try:
        summarizer = Summarizer(
            llm, db, fetcher, topic_text(topic),
            model_key=f"{cfg.rerank.provider}/{cfg.rerank.model}",
            max_input_tokens=cfg.summarize.max_input_tokens,
            topic_slug=topic.slug,
        )
        summaries = summarizer.summarize_all([it.ranked.paper for it in selected])
    finally:
        if own:
            fetcher.close()
    log.info("PDF downloads: %d, summary cache hits: %d", fetcher.downloads - downloads_before, summarizer.cache_hits)
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


def save_digest(text: str, out_dir: Path, run_id: int, slug: str = "") -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{_stamp(slug, run_id) if slug else f'{datetime.now(timezone.utc):%Y-%m-%d}_run{run_id}'}.md"
    path.write_text(text, encoding="utf-8")
    return path


# --- one run, many topics ---
@dataclass
class Shared:
    """Created once per run and shared by every topic."""

    cfg: Config
    args: argparse.Namespace
    db: Database
    timer: StageTimer
    fetcher: PdfFetcher
    arxiv_client: object | None = None
    _llm: BudgetedLLM | None = None
    _llm_tried: bool = False
    _gh: object | None = None
    _gh_tried: bool = False

    def llm(self) -> BudgetedLLM | None:
        """One LLM client for every topic: one key, one call counter, one budget."""
        if not self._llm_tried:
            self._llm_tried = True
            inner = make_llm(self.cfg.rerank.provider, self.cfg.rerank.model)
            self._llm = BudgetedLLM(inner, self.cfg.llm.max_calls_per_run) if inner is not None else None
        return self._llm

    def github(self):
        if not self._gh_tried:
            self._gh_tried = True
            self._gh = make_github()
        return self._gh

    def close(self) -> None:
        self.fetcher.close()
        if self._gh is not None:
            self._gh.close()


@dataclass
class TopicOutcome:
    slug: str
    run_id: int | None = None
    problems: list[str] = field(default_factory=list)
    failed: bool = False  # crashed or couldn't fetch: nothing was produced for this topic
    rerank_calls: int = 0
    summarize_calls: int = 0
    selected: int = 0
    issue: int | None = None
    seconds: float = 0.0
    printouts: list = field(default_factory=list)  # (kind, payload) printed after all topics


def run_topic(shared: Shared, topic: TopicConfig) -> TopicOutcome:
    cfg, args, db, timer = shared.cfg, shared.args, shared.db, shared.timer
    out = TopicOutcome(topic.slug)
    rerank_on = cfg.rerank.enabled and not args.no_rerank
    summarize_on = cfg.summarize.enabled and not args.no_summarize
    log.info("=== Topic: %s (%s) ===", topic.name, topic.slug)

    # 1. fetch
    with timer("fetch"):
        if args.sample:
            papers = load_sample(args.sample)
        else:
            try:
                papers = fetch_recent(topic.keywords, topic.categories, cfg.fetch.days_back, cfg.fetch.max_results,
                                      client=shared.arxiv_client)
            except Exception as e:  # network down, arXiv overloaded, etc.
                log.error("[%s] Could not reach arXiv (%s: %s). Check your internet, or test offline with "
                          "--sample tests/fixtures/sample_papers.json", topic.slug, type(e).__name__, str(e)[:200])
                out.failed = True
                out.problems.append(f"arXiv fetch failed ({type(e).__name__})")
                return out

    # 2. dedup, per topic: another topic having seen a paper doesn't count
    new = papers if args.include_seen else db.filter_new(papers, topic.slug)
    log.info("%d fetched, %d new (%d already seen by this topic)", len(papers), len(new), len(papers) - len(new))

    # 3. hybrid rank
    with timer("rank"):
        ranker, embedder = build_ranker(cfg, db, use_embeddings=not args.no_embeddings, topic=topic)
        log.info("Retrievers: %s", ", ".join(ranker.names))
        ranked = ranker.rank(topic.ranking_query(), new, cfg.ranking.top_k)
    if embedder is not None:
        log.info("Gemini embedding API calls: %d", embedder.api_calls)
    out.printouts.append(("ranking", ranked))

    llm = shared.llm() if ranked and (rerank_on or summarize_on) else None
    calls0 = llm.calls if llm is not None else 0

    # 4. LLM rerank (falls back to hybrid order on any failure)
    with timer("rerank"):
        result = run_rerank(cfg, ranked, llm, enabled=rerank_on, topic=topic)
    out.rerank_calls = (llm.calls - calls0) if llm is not None else 0
    log.info(result.note)
    if rerank_on and result.items and not result.used_llm:
        out.problems.append(f"LLM rerank fell back to hybrid order ({result.note})")

    # 5. summarize the selected papers (falls back to the abstract on any failure)
    summaries: list[PaperSummary] = []
    if summarize_on and result.top:
        calls1 = llm.calls if llm is not None else 0
        with timer("summarize"):
            summaries = run_summarize(cfg, db, result.top, llm, topic=topic, fetcher=shared.fetcher)
        out.summarize_calls = (llm.calls - calls1) if llm is not None else 0
        log.info(summaries_note(summaries))
        if all(s.summary is None for s in summaries):
            out.problems.append("no paper got a summary")
    elif result.top:
        log.info("Summaries off: the digest shows abstracts")
        summaries = [PaperSummary(it.ranked.paper, None, "abstract", error="summaries off") for it in result.top]
    out.selected = len(result.top)

    # 6. record the run. On --dry-run this goes to the throwaway in-memory DB, so the digest
    #    and the GitHub preview below are built by exactly the code a real run uses.
    notes = result.note + (f"; {summaries_note(summaries)}" if summarize_on and summaries else "")
    db.save_papers(new, topic.slug)
    run_id = db.log_run(len(papers), len(new), len(ranked), ranker.active, notes=notes, topic=topic.slug)
    out.run_id = run_id
    db.save_rankings(run_id, (
        {"paper_id": it.ranked.paper.id, "hybrid_rank": it.hybrid_rank, "hybrid_score": it.ranked.score,
         "llm_score": it.llm_score, "reason": it.reason, "final_rank": it.final_rank}
        for it in result.items
    ))

    # 7. files (saved before printing, so a console problem can never lose a paid-for run)
    if args.dry_run:
        log.info("Dry run: nothing written (no database changes, no candidates or digest file)")
    else:
        log.info("Run #%d logged (%s). Papers seen by this topic: %d", run_id, topic.slug,
                 db.count_papers(topic.slug))
        if result.items:
            path = save_candidates(result, cfg.db_path.parent / "candidates", run_id, topic.slug)
            log.info("Saved candidates to %s", path)
        if result.top:
            ranked_by = f"{llm.name}/{llm.model}" if result.used_llm and llm is not None else "hybrid score"
            text = render_digest(load_entries(db, cfg, run_id), topic.name, run_id,
                                 datetime.now(timezone.utc).date(), ranked_by)
            path = save_digest(text, cfg.db_path.parent / "digests", run_id, topic.slug)
            log.info("Saved digest to %s", path)

    # 8. deliver to GitHub (a failure is logged with a resume command; the digest file is already saved)
    deliver_on = cfg.deliver.provider == "github" and not args.no_deliver and bool(result.top)
    if deliver_on and args.dry_run:
        out.printouts.append(("preview", preview(build_content(db, cfg, run_id))))
    elif deliver_on:
        with timer("deliver"):
            gh = shared.github()
            if gh is None:
                out.problems.append("GitHub delivery skipped: no GITHUB_TOKEN / GITHUB_REPO")
            else:
                delivery = deliver_run(db, cfg, run_id, gh)
                log_result(delivery)
                out.issue = delivery.issue_number
                if delivery.error:
                    out.problems.append(f"GitHub delivery failed ({delivery.error})")
    elif result.top:
        log.info("Delivery off (%s)", "--no-deliver" if args.no_deliver else f"provider: {cfg.deliver.provider}")

    out.printouts += [("rerank", result), ("summaries", summaries)]
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Research Digest Agent pipeline")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--topic", help="run only this topic (slug from config.yaml); default: every topic")
    ap.add_argument("--sample", help="load papers from a JSON file instead of arXiv (offline)")
    ap.add_argument("--no-embeddings", action="store_true", help="BM25 only, no Gemini calls")
    ap.add_argument("--no-rerank", action="store_true", help="skip LLM reranking, keep the hybrid top N")
    ap.add_argument("--no-summarize", action="store_true", help="skip PDF download and summaries")
    ap.add_argument("--no-deliver", action="store_true", help="don't post the digest to GitHub")
    ap.add_argument("--include-seen", action="store_true", help="rank papers even if seen in an earlier run")
    ap.add_argument("--dry-run", action="store_true",
                    help="write nothing: no DB changes, no output files; print the GitHub issue instead of posting")
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 if any topic failed or fell back (LLM, GitHub); for CI, so failures get noticed")
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
    topics = [cfg.get_topic(args.topic)] if args.topic else cfg.topics
    log.info("Topics: %s", ", ".join(t.slug for t in topics))
    timer = StageTimer()
    outcomes: list[TopicOutcome] = []

    arxiv_client = None
    if not args.sample:
        import arxiv

        # one client for all topics, so arXiv's "1 request per 3 seconds" holds across them
        arxiv_client = arxiv.Client(page_size=100, delay_seconds=3.0, num_retries=3)

    # --dry-run works on an in-memory snapshot: same reads as a real run, nothing written to disk
    with Database.for_config(cfg, in_memory_copy=args.dry_run) as db:
        shared = Shared(cfg, args, db, timer, PdfFetcher(delay_seconds=cfg.summarize.pdf_delay_seconds),
                        arxiv_client)
        try:
            for topic in topics:
                t0 = time.perf_counter()
                try:
                    outcome = run_topic(shared, topic)
                except Exception as e:  # one topic's bug or outage must not cost the others their digest
                    log.exception("[%s] Topic failed: %s: %s", topic.slug, type(e).__name__, e)
                    outcome = TopicOutcome(topic.slug, failed=True,
                                           problems=[f"crashed ({type(e).__name__}: {str(e)[:150]})"])
                outcome.seconds = time.perf_counter() - t0
                outcomes.append(outcome)
        finally:
            shared.close()

        llm = shared._llm
        for o in outcomes:
            status = "FAILED" if o.failed else ("degraded" if o.problems else "ok")
            log.info("Topic %s: %s, %d papers, LLM calls %d (rerank %d, summarize %d)%s, %.0fs", o.slug, status,
                     o.selected, o.rerank_calls + o.summarize_calls, o.rerank_calls, o.summarize_calls,
                     f", issue #{o.issue}" if o.issue else "", o.seconds)
        if llm is not None:
            log.info("LLM calls this run: %d of a %d budget%s", llm.calls, llm.max_calls,
                     f" | Groq quota: {describe_rate_limits(llm.rate_limits)}" if llm.rate_limits else "")
        log.info("Timing: %s", timer.line())

        for o in outcomes:  # printed last, after everything is saved
            for kind, payload in o.printouts:
                if kind == "ranking":
                    print(f"\n##### {o.slug} #####")
                    print_ranking(payload)
                elif kind == "preview":
                    print("\nGitHub issue preview (dry run: nothing posted)")
                    print(payload)
                elif kind == "rerank":
                    print_rerank(payload)
                elif kind == "summaries":
                    print_summaries(payload)

    problems = [f"{o.slug}: {p}" for o in outcomes for p in o.problems]
    if problems:
        # everything that worked is already saved; this only decides the exit code
        level = logging.ERROR if args.strict else logging.WARNING
        for p in problems:
            log.log(level, "Degraded run: %s", p)
    if outcomes and all(o.failed for o in outcomes):
        return 1  # nothing worked at all
    if problems and args.strict:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
