"""Compare ranking methods on your hand labels: Precision@5, Recall@20, NDCG@10.

    python -m digest.eval                  # all methods; LLM scores cached in data/labels.db
    python -m digest.eval --no-llm         # skip anything that would call the LLM (cache still used)

Protocol (2-fold cross-validation, the same for every method):
- The labelled papers are split into two halves, stratified by label, in a fixed order.
- Each method ranks one half at a time against the topic query; metrics are averaged over halves.
- Only the feedback hybrid has a parameter (its weight). It is chosen on one half and reported on
  the other: to report half A, the feedback retriever is trained on half B's labels (+ 👍/👎 on
  papers outside A), and the weight is the one that scored best on B (trained on B with each
  paper's own label left out). Then the halves swap. The reported half never sees its own labels.
- LLM rerank = the production setup: BM25 order, top 20 reordered by LLM score.

The report is printed and saved to data/evals/<date>.md.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import statistics
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

from digest.config import Config, load_config
from digest.labels import LabelStore, shuffle_key
from digest.llm import LLMAuthError, LLMClient
from digest.metrics import ndcg_at_k, precision_at_k, recall_at_k
from digest.models import Paper
from digest.prompting import LLMFailed
from digest.rerank import PROMPTS, score_batch, system_prompt
from digest.retrieval import BM25Retriever, HybridRanker, RankedPaper
from digest.retrieval.feedback import Example, FeedbackRetriever
from digest.retrieval.tfidf import TfidfRetriever

log = logging.getLogger(__name__)

RERANK_DEPTH = 20  # production reranks the hybrid top 20
WEIGHT_GRID = (0.2, 0.35, 0.5)
METRICS = ("P@5", "R@20", "NDCG@10")


# --- data ---
@dataclass
class Dataset:
    papers: list[Paper]
    relevant: dict[str, int]  # paper_id -> 0/1
    reactions: list[Example] = field(default_factory=list)
    previous_round: dict[str, int] = field(default_factory=dict)  # last archived labelling round, if any

    @property
    def n_relevant(self) -> int:
        return sum(self.relevant.values())


def two_folds(data: Dataset, seed: int = 0) -> tuple[list[Paper], list[Paper]]:
    """Stratified, deterministic: alternate papers of each class between the halves.
    seed 0 is the canonical split; other seeds give other (still reproducible) splits."""
    def key(p: Paper) -> str:
        return shuffle_key(p.id) if seed == 0 else hashlib.sha1(f"split-{seed}:{p.id}".encode()).hexdigest()

    a, b = [], []
    for label in (1, 0):
        group = sorted((p for p in data.papers if data.relevant[p.id] == label), key=key)
        for i, p in enumerate(group):
            (a if i % 2 == 0 else b).append(p)
    return a, b


def evaluate_ranking(ranking: Sequence[str], relevant: dict[str, int]) -> dict[str, float]:
    return {
        "P@5": precision_at_k(ranking, relevant, 5),
        "R@20": recall_at_k(ranking, relevant, 20),
        "NDCG@10": ndcg_at_k(ranking, relevant, 10),
    }


# --- rankers: each takes the papers of one half and returns ids, best first ---
def _order(papers: Sequence[Paper], scores: Sequence[float]) -> list[str]:
    return [p.id for p, _ in sorted(zip(papers, scores), key=lambda x: (-x[1], x[0].id))]


def rank_bm25(query: str, papers: Sequence[Paper]) -> list[str]:
    return _order(papers, BM25Retriever().score(query, papers))


def rank_tfidf(query: str, papers: Sequence[Paper]) -> list[str]:
    return _order(papers, TfidfRetriever().score(query, papers))


def rank_rerank(query: str, papers: Sequence[Paper], llm_scores: dict[str, float],
                base: list[str] | None = None) -> list[str]:
    """A base order (default BM25), then its top RERANK_DEPTH reordered by LLM score (ties: base order);
    unscored papers keep their place after the scored ones."""
    base = base if base is not None else rank_bm25(query, papers)
    head, tail = base[:RERANK_DEPTH], base[RERANK_DEPTH:]
    pos = {pid: i for i, pid in enumerate(head)}
    scored = sorted((pid for pid in head if pid in llm_scores), key=lambda pid: (-llm_scores[pid], pos[pid]))
    unscored = [pid for pid in head if pid not in llm_scores]
    return scored + unscored + tail


def rank_feedback(query: str, papers: Sequence[Paper], examples: Sequence[Example], weight: float) -> list[str]:
    fb = FeedbackRetriever(examples)
    ranked = HybridRanker([(BM25Retriever(), 1.0 - weight), (fb, weight)]).rank(query, papers, len(papers))
    return [rp.paper.id for rp in ranked]


def label_examples(papers: Sequence[Paper], relevant: dict[str, int]) -> list[Example]:
    return [Example(p, 1 if relevant[p.id] else -1, "label") for p in papers]


# --- LLM scores (cached per paper, model and prompt version) ---
def llm_scores_for(version: str, papers: Sequence[Paper], store: LabelStore, model_key: str, topic: str,
                   llm: LLMClient | None, batch_size: int = 5) -> dict[str, float]:
    """Raw scores on the prompt's own scale. Missing ones are fetched in a fixed order, if llm is given."""
    prompt = PROMPTS[version]
    cached = {pid: s for pid, (s, _) in store.get_llm_scores(model_key, version).items()}
    todo = sorted((p for p in papers if p.id not in cached), key=lambda p: shuffle_key(p.id))
    if todo and llm is None:
        log.warning("%s: %d papers have no cached score and no LLM is available", version, len(todo))
    elif todo:
        log.info("%s: scoring %d papers (%d cached)", version, len(todo), len(cached))
        system = system_prompt(topic, version)
        for i in range(0, len(todo), batch_size):
            batch = todo[i : i + batch_size]
            try:
                items = score_batch(llm, system, [RankedPaper(p, 0.0) for p in batch], prompt)
            except LLMAuthError:
                raise
            except LLMFailed as e:
                log.warning("%s: batch failed (%s); those papers stay unscored", version, e)
                continue
            for idx, item in items.items():
                store.put_llm_score(batch[idx].id, model_key, version, item.score, item.reason)
                cached[batch[idx].id] = item.score
    return {p.id: cached[p.id] for p in papers if p.id in cached}


# --- the evaluation ---
@dataclass
class Row:
    method: str
    folds: list[dict[str, float]]
    note: str = ""

    @property
    def mean(self) -> dict[str, float]:
        return {m: statistics.fmean(f[m] for f in self.folds) for m in METRICS}


@dataclass
class EvalResult:
    data: Dataset
    folds: tuple[list[Paper], list[Paper]]
    rows: list[Row]
    raw_scores: dict[str, dict[str, float]]  # version -> paper -> raw score
    chosen_weights: list[float]
    weight_table: list[dict[float, float]]  # per fold: weight -> NDCG@10 on the choosing half

    def best_prompt(self) -> str | None:
        rerank_rows = [r for r in self.rows if r.method.startswith("BM25 + LLM rerank")]
        if not rerank_rows:
            return None
        best = max(rerank_rows, key=lambda r: (r.mean["NDCG@10"], -int(r.method.split()[-1][1:])))
        return best.method.split()[-1]


def run_eval(data: Dataset, query: str, llm_scores: dict[str, dict[str, float]],
             weights: Sequence[float] = WEIGHT_GRID, seed: int = 0) -> EvalResult:
    folds = two_folds(data, seed)
    rel = data.relevant
    rows: dict[str, Row] = {}

    def add(method: str, fold_metrics: dict[str, float], note: str = "") -> None:
        rows.setdefault(method, Row(method, [], note)).folds.append(fold_metrics)

    chosen, weight_table = [], []
    for test, other in (folds, folds[::-1]):
        test_rel = {p.id: rel[p.id] for p in test}
        add("BM25 (current)", evaluate_ranking(rank_bm25(query, test), test_rel))
        add("TF-IDF", evaluate_ranking(rank_tfidf(query, test), test_rel))
        for version, scores in llm_scores.items():
            add(f"BM25 + LLM rerank {version}", evaluate_ranking(rank_rerank(query, test, scores), test_rel),
                PROMPTS[version].description)

        # feedback: choose the weight on `other` (leave-one-out inside it), report on `test`
        test_ids = {p.id for p in test}
        other_rel = {p.id: rel[p.id] for p in other}
        reactions = [e for e in data.reactions if e.paper.id not in test_ids]
        train = reactions + label_examples(other, rel)
        by_w = {w: ndcg_at_k(rank_feedback(query, other, train, w), other_rel, 10) for w in weights}
        best_w = max(by_w, key=lambda w: (by_w[w], -w))  # ties: the smaller, more conservative weight
        chosen.append(best_w)
        weight_table.append(by_w)
        fb_order = rank_feedback(query, test, train, best_w)
        add("BM25 + feedback (weight by 2-fold CV)", evaluate_ranking(fb_order, test_rel))
        for version, scores in llm_scores.items():  # what production would run: hybrid top 20 -> LLM
            add(f"BM25 + feedback + LLM rerank {version}",
                evaluate_ranking(rank_rerank(query, test, scores, base=fb_order), test_rel), "production combination")

    return EvalResult(data, folds, list(rows.values()), llm_scores, chosen, weight_table)


# --- repeated cross-validation: one split is too noisy to compare methods ---
@dataclass
class Repeated:
    results: list[EvalResult]

    def methods(self) -> list[str]:
        return [r.method for r in self.results[0].rows]

    def values(self, method: str, metric: str) -> list[float]:
        return [next(r for r in res.rows if r.method == method).mean[metric] for res in self.results]

    def summary(self, method: str, metric: str) -> tuple[float, float]:
        v = self.values(method, metric)
        return statistics.fmean(v), (statistics.stdev(v) if len(v) > 1 else 0.0)

    def wins(self, method: str, baseline: str, metric: str = "NDCG@10") -> int:
        """Splits where `method` beats `baseline` (ties don't count)."""
        return sum(a > b for a, b in zip(self.values(method, metric), self.values(baseline, metric)))

    def best_prompt(self) -> str | None:
        rerank = [m for m in self.methods() if m.startswith("BM25 + LLM rerank")]
        if not rerank:
            return None
        best = max(rerank, key=lambda m: (self.summary(m, "NDCG@10")[0], -int(m.split()[-1][1:])))
        return best.split()[-1]


def run_repeated(data: Dataset, query: str, llm_scores: dict[str, dict[str, float]], repeats: int = 20,
                 weights: Sequence[float] = WEIGHT_GRID) -> Repeated:
    return Repeated([run_eval(data, query, llm_scores, weights, seed) for seed in range(repeats)])


def render_repeated(rep: Repeated, baseline: str = "BM25 (current)") -> str:
    n = len(rep.results)
    d = rep.results[0].data
    rand = [statistics.fmean(random_baseline(h, d.relevant)[m] for h in res.folds) for res in rep.results
            for m in ("NDCG@10",)]
    rand_p5 = statistics.fmean(d.relevant[p.id] for p in d.papers)
    notes = {r.method: r.note for r in rep.results[0].rows}
    lines = [
        f"## Main result: {n} repeated 2-fold splits",
        "",
        f"Each split is a different stratified 2-fold partition; every method is scored on both halves and "
        f"averaged, then the {n} splits give a mean ± standard deviation. \"beats BM25\" counts the splits "
        "where the method's NDCG@10 is higher than BM25's on the same split.",
        "",
        "| Method | P@5 | R@20 | NDCG@10 | beats BM25 | Notes |",
        "|---|---|---|---|---|---|",
        f"| *Random order (expected)* | {rand_p5:.2f} | | {statistics.fmean(rand):.3f} | | the bar to beat |",
    ]
    for m in rep.methods():
        cells = []
        for metric in METRICS:
            mean, sd = rep.summary(m, metric)
            cells.append(f"{mean:.2f} ± {sd:.2f}" if metric != "NDCG@10" else f"{mean:.3f} ± {sd:.3f}")
        wins = "-" if m == baseline else f"{rep.wins(m, baseline)}/{n}"
        lines.append(f"| {m} | {' | '.join(cells)} | {wins} | {notes.get(m, '')} |")
    weights = Counter(w for res in rep.results for w in res.chosen_weights)
    lines += ["", "Feedback weight chosen by the inner CV, across all halves: "
              + ", ".join(f"{w:g} ({c}x)" for w, c in sorted(weights.items())) + ".", ""]
    best = rep.best_prompt()
    if best:
        lines += [f"**Best rerank prompt by mean NDCG@10: {best}.**", ""]
    return "\n".join(lines)


# --- report ---
def score_distribution(version: str, raw: dict[str, float], relevant: dict[str, int]) -> dict:
    prompt = PROMPTS[version]
    tens = [prompt.to_ten(s) for s in raw.values()]
    top_cut = 9 if prompt.hi == 10 else 90
    hist = Counter(round(t) for t in tens)
    return {
        "n": len(raw),
        "hist": {k: hist.get(k, 0) for k in range(1, 11)},
        "share_top": sum(1 for s in raw.values() if s >= top_cut) / len(raw) if raw else 0.0,
        "top_cut": top_cut,
        "distinct": len(set(raw.values())),
        "auc": auc(raw, relevant),
        "mean_rel": statistics.fmean(prompt.to_ten(raw[p]) for p in raw if relevant.get(p)) if any(
            relevant.get(p) for p in raw) else float("nan"),
        "mean_not": statistics.fmean(prompt.to_ten(raw[p]) for p in raw if not relevant.get(p)) if any(
            not relevant.get(p) for p in raw) else float("nan"),
    }


def auc(scores: dict[str, float], relevant: dict[str, int]) -> float:
    """Probability a random relevant paper outscores a random non-relevant one (ties count half)."""
    pos = [s for p, s in scores.items() if relevant.get(p)]
    neg = [s for p, s in scores.items() if not relevant.get(p)]
    if not pos or not neg:
        return float("nan")
    wins = sum(1.0 if a > b else 0.5 if a == b else 0.0 for a in pos for b in neg)
    return wins / (len(pos) * len(neg))


def random_baseline(papers: Sequence[Paper], relevant: dict[str, int]) -> dict[str, float]:
    """Expected metrics of a uniformly random ordering, exactly (no simulation).

    E[P@k] = base rate; E[R@k] = k / n; E[DCG@k] = base rate * sum of discounts, divided by the ideal.
    """
    import math

    n = len(papers)
    n_rel = sum(relevant[p.id] for p in papers)
    rate = n_rel / n if n else 0.0
    ideal = sum(1 / math.log2(i + 2) for i in range(min(10, n_rel)))
    expected_dcg = rate * sum(1 / math.log2(i + 2) for i in range(min(10, n)))
    return {"P@5": rate, "R@20": min(20, n) / n if n else 0.0, "NDCG@10": expected_dcg / ideal if ideal else 0.0}


def cohen_kappa(a: dict[str, int], b: dict[str, int]) -> tuple[float, int]:
    """Agreement between two labelling rounds on the papers both labelled, corrected for chance."""
    common = [k for k in a if k in b]
    if not common:
        return float("nan"), 0
    n = len(common)
    observed = sum(a[k] == b[k] for k in common) / n
    pa, pb = sum(a[k] for k in common) / n, sum(b[k] for k in common) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    return ((observed - expected) / (1 - expected) if expected < 1 else 1.0), n


def render_report(r: EvalResult, model_key: str, today: str) -> str:
    d = r.data
    a, b = r.folds
    lines = [
        f"# Ranking evaluation, {today}",
        "",
        f"**Dataset:** {len(d.papers)} hand-labelled papers ({d.n_relevant} relevant, "
        f"{len(d.papers) - d.n_relevant} not), from a one-off 30-day arXiv fetch with the topic keywords. "
        f"2-fold cross-validation, stratified: halves of {len(a)} and {len(b)} papers. "
        f"Every method ranks one half at a time; numbers are the mean of the two halves.",
        "",
        "| Method | P@5 | R@20 | NDCG@10 | Notes |",
        "|---|---|---|---|---|",
    ]
    rb = [random_baseline(half, d.relevant) for half in r.folds]
    rm = {k: statistics.fmean(x[k] for x in rb) for k in METRICS}
    lines.append(f"| *Random order (expected)* | {rm['P@5']:.2f} | {rm['R@20']:.2f} | {rm['NDCG@10']:.3f} | "
                 "the bar to beat |")
    for row in r.rows:
        m = row.mean
        lines.append(f"| {row.method} | {m['P@5']:.2f} | {m['R@20']:.2f} | {m['NDCG@10']:.3f} | {row.note} |")
    lines += [
        "",
        "- R@20 is the same for every rerank row by construction: reranking only reorders BM25's top 20.",
        *(f"- Label consistency: Cohen's kappa {k:.2f} between this round and the previous labelling round "
          f"on {n} papers labelled in both (1 = identical, 0 = chance)."
          for k, n in [cohen_kappa(d.relevant, d.previous_round)] if n),
        f"- Feedback weight chosen per half: {', '.join(f'{w:g}' for w in r.chosen_weights)} "
        f"(grid {', '.join(f'{w:g}' for w in r.weight_table[0])}; NDCG@10 on the choosing half: "
        + "; ".join(", ".join(f"{w:g} -> {v:.3f}" for w, v in t.items()) for t in r.weight_table) + ").",
        f"- Per half, P@5 moves in steps of 0.2 (one paper), so differences under ~0.1 are noise.",
        "",
    ]
    if r.raw_scores:
        lines += ["## LLM score distribution (score bunching)", "",
                  f"All {len(d.papers)} labelled papers scored by `{model_key}`, shown on a 1-10 scale.", "",
                  "| Prompt | ≤3 | 4-5 | 6-7 | 8 | 9 | 10 | top band | distinct values | mean (rel / not) | AUC |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for version, raw in r.raw_scores.items():
            s = score_distribution(version, raw, d.relevant)
            h = s["hist"]
            lines.append(
                f"| {version} | {sum(h[k] for k in (1, 2, 3))} | {h[4] + h[5]} | {h[6] + h[7]} | {h[8]} | {h[9]} | "
                f"{h[10]} | {s['share_top']:.0%} (≥{s['top_cut']}) | {s['distinct']} | "
                f"{s['mean_rel']:.1f} / {s['mean_not']:.1f} | {s['auc']:.2f} |")
        lines += ["", "AUC: probability that a random relevant paper gets a higher LLM score than a random "
                  "non-relevant one (0.5 = no better than chance).", ""]
        best = r.best_prompt()
        if best:
            lines += [f"**Best rerank prompt by NDCG@10: {best}.**", ""]
    return "\n".join(lines)


# --- CLI ---
def load_dataset(cfg: Config) -> Dataset:
    from digest.db import Database

    with LabelStore(cfg.labels_path, read_only=True) as store:
        labelled = store.labelled_papers()
        rounds = store.conn.execute("SELECT MAX(round) FROM labels_archive").fetchone()[0]
        previous = store.archived_labels(rounds) if rounds else {}
    reactions = []
    if cfg.db_path.exists():
        with Database.for_config(cfg, in_memory_copy=True) as db:  # never writes digest.db
            for row in db.conn.execute("SELECT paper_id, rating FROM feedback WHERE source = 'github'"):
                paper = db.get_paper(row["paper_id"])
                if paper is not None:
                    reactions.append(Example(paper, int(row["rating"]), "reaction"))
    return Dataset([p for p, _ in labelled], {p.id: r for p, r in labelled}, reactions, previous)


def main(argv: list[str] | None = None, make_llm: Callable | None = None) -> int:
    from dotenv import load_dotenv

    ap = argparse.ArgumentParser(description="Evaluate ranking methods on your hand labels")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--versions", nargs="*", default=list(PROMPTS), help="rerank prompt versions to compare")
    ap.add_argument("--no-llm", action="store_true", help="never call the LLM (cached scores are still used)")
    ap.add_argument("--out", help="report path (default data/evals/<date>.md)")
    ap.add_argument("--repeats", type=int, default=20, help="random 2-fold splits for the main table")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    for noisy in ("httpx", "groq"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    load_dotenv()
    cfg = load_config(args.config)

    data = load_dataset(cfg)
    if len(data.papers) < 10 or data.n_relevant == 0 or data.n_relevant == len(data.papers):
        log.error("Need at least 10 labelled papers with both classes (have %d, %d relevant). "
                  "Run python -m digest.label first.", len(data.papers), data.n_relevant)
        return 1

    model_key = f"{cfg.rerank.provider}/{cfg.rerank.model}"
    topic = f"{cfg.topic.name}. {cfg.topic.description}"
    llm = None
    if not args.no_llm:
        if make_llm is None:
            from digest.llm import make_llm
        llm = make_llm(cfg.rerank.provider, cfg.rerank.model)
    scores: dict[str, dict[str, float]] = {}
    with LabelStore(cfg.labels_path) as store:
        for version in args.versions:
            got = llm_scores_for(version, data.papers, store, model_key, topic, llm)
            if len(got) < len(data.papers) * 0.9:
                log.warning("%s: only %d/%d papers scored, left out of the comparison", version, len(got),
                            len(data.papers))
                continue
            scores[version] = got
    if llm is not None:
        log.info("LLM calls this eval: %d", llm.calls)

    query = cfg.topic.ranking_query()
    rep = run_repeated(data, query, scores, repeats=max(1, args.repeats))
    result = rep.results[0]  # the canonical split: per-half details and score distributions
    today = f"{datetime.now(timezone.utc):%Y-%m-%d}"
    canonical = render_report(result, model_key, today).split("\n", 2)
    report = "\n".join([canonical[0], canonical[1], render_repeated(rep),
                        "## Details: the canonical split", "", canonical[2]])
    out = Path(args.out) if args.out else cfg.db_path.parent / "evals" / f"{today}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(report)
    log.info("Saved %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
