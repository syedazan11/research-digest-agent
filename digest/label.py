"""Label papers as relevant / not relevant, one at a time, for evaluation and feedback learning.

    python -m digest.label --fetch          # one-off: pull ~30 days of papers into the label pool
    python -m digest.label                  # label: y = relevant, n = not, s = skip, u = undo, q = quit
    python -m digest.label --stats

Labels are saved the moment you press a key, so quitting and coming back resumes where you left off.
Papers come in a fixed shuffled order with no rank or score shown, so the current ranker can't
anchor your judgement. Everything lives in data/labels.db; data/digest.db is never touched.
"""

from __future__ import annotations

import argparse
import logging
import sys
import textwrap
from typing import Callable

from digest.labels import LabelStore
from digest.models import Paper

log = logging.getLogger(__name__)

HELP = "[y] relevant  [n] not relevant  [s] skip  [u] undo  [q] quit"
WIDTH = 100


def read_key() -> str:
    """One keypress, no Enter needed (falls back to a line of input where that isn't possible)."""
    try:
        import msvcrt  # Windows

        return msvcrt.getwch()
    except ImportError:
        pass
    try:
        import termios
        import tty

        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            return sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
    except Exception:
        return (input("> ").strip() or " ")[0]


def render(paper: Paper, position: int, total: int, done: int, rubric: str = "") -> str:
    authors = ", ".join(paper.authors[:4]) + (" et al." if len(paper.authors) > 4 else "")
    lines = [
        "=" * WIDTH,
        *([rubric, "-" * WIDTH] if rubric else []),
        f"[{position}/{total}]  labelled so far: {done}",
        "",
        *textwrap.wrap(paper.title, WIDTH),
        f"{authors}  |  {paper.published:%Y-%m-%d}  |  {', '.join(paper.categories[:4])}",
        "",
        *textwrap.wrap(paper.abstract, WIDTH),
        "",
        HELP,
    ]
    return "\n".join(lines)


def label_loop(store: LabelStore, key: Callable[[], str] = read_key, out: Callable[[str], None] = print,
               rubric: str = "") -> dict:
    todo = store.unlabelled()
    total_pool = store.stats()["pool"]
    already = total_pool - len(todo)
    history: list[str] = []  # paper ids labelled this session, for undo
    counts = {"relevant": 0, "not_relevant": 0, "skipped": 0, "undone": 0}
    i = 0
    if not todo:
        out("Nothing to label. Fetch a pool first: python -m digest.label --fetch")
        return counts
    while i < len(todo):
        paper = todo[i]
        out(render(paper, already + i + 1, total_pool, already + len(history), rubric))
        k = (key() or " ").lower()
        if k in ("y", "n"):
            store.set_label(paper.id, k == "y")
            history.append(paper.id)
            counts["relevant" if k == "y" else "not_relevant"] += 1
            i += 1
        elif k == "s":
            counts["skipped"] += 1
            i += 1
        elif k == "u":
            if not history:
                out("(nothing to undo)")
                continue
            last = history.pop()
            store.remove_label(last)
            counts["undone"] += 1
            i = next(j for j, p in enumerate(todo) if p.id == last)
            out("(undone)")
        elif k in ("q", "\x03", "\x1b"):  # q, Ctrl+C, Esc
            break
        else:
            out(HELP)
    return counts


def main(argv: list[str] | None = None) -> int:
    from digest.config import load_config
    from digest.fetch import fetch_recent

    ap = argparse.ArgumentParser(description="Label papers for evaluation and feedback learning")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--fetch", action="store_true", help="add the last --days of arXiv papers to the pool")
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--max", type=int, default=300, help="max papers to fetch")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--relabel", action="store_true",
                    help="start a fresh round: current labels move to labels_archive; pool + LLM cache stay")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    logging.getLogger("arxiv").setLevel(logging.WARNING)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    cfg = load_config(args.config)

    with LabelStore(cfg.labels_path) as store:
        if args.relabel:
            rnd, n = store.archive_labels()
            print(f"Archived {n} labels as round {rnd} (labels_archive). The pool and cached LLM scores are kept.")
        if args.fetch:
            papers = fetch_recent(cfg.topic.keywords, cfg.topic.categories, args.days, args.max)
            added = store.add_to_pool(papers)
            log.info("Label pool: %d fetched, %d new. Only data/labels.db was changed.", len(papers), added)
        elif not args.stats:
            counts = label_loop(store, rubric=cfg.topic.label_rubric)
            print(f"\nThis session: {counts['relevant']} relevant, {counts['not_relevant']} not, "
                  f"{counts['skipped']} skipped, {counts['undone']} undone")
        s = store.stats()
        print(f"Pool {s['pool']}: {s['relevant']} relevant, {s['not_relevant']} not relevant, "
              f"{s['unlabelled']} unlabelled  ({cfg.labels_path})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
