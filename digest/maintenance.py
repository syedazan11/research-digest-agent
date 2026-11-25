"""Small DB chores for the GitHub Actions workflow, kept in Python so they're testable.

    python -m digest.maintenance last-run                  # newest run id (0 if none)
    python -m digest.maintenance vacuum                    # compact the DB before committing it
    python -m digest.maintenance commit-message --since 4  # "digest: run 5 (5 papers, issue #6)"
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

from digest.config import load_config


def last_run_id(db_path: Path) -> int:
    if not db_path.exists():
        return 0
    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT MAX(id) FROM runs").fetchone()
    return int(row[0] or 0)


def vacuum(db_path: Path) -> tuple[int, int]:
    """Rewrite the DB without free pages. Returns (bytes before, bytes after)."""
    before = db_path.stat().st_size
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("VACUUM")
    finally:
        conn.close()
    return before, db_path.stat().st_size


def commit_message(db_path: Path, since_run: int) -> str:
    """What the bot's commit says: the run(s) logged since `since_run`, or that only feedback changed."""
    with sqlite3.connect(db_path) as conn:
        runs = conn.execute("SELECT id FROM runs WHERE id > ? ORDER BY id", (since_run,)).fetchall()
        if not runs:
            n = conn.execute("SELECT COUNT(*) FROM feedback WHERE source = 'github'").fetchone()[0]
            return f"digest: feedback only ({n} ratings)"
        run_id = runs[-1][0]
        n_papers = conn.execute("SELECT COUNT(*) FROM rankings WHERE run_id = ? AND selected = 1",
                                (run_id,)).fetchone()[0]
        issue = conn.execute("SELECT issue_number FROM digest_issues WHERE run_id = ?", (run_id,)).fetchone()
    parts = [f"{n_papers} paper{'s' if n_papers != 1 else ''}"]
    if issue is not None:
        parts.append(f"issue #{issue[0]}")
    return f"digest: run {run_id} ({', '.join(parts)})"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="DB chores for the weekly workflow")
    ap.add_argument("--config", default="config.yaml")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("last-run")
    sub.add_parser("vacuum")
    cm = sub.add_parser("commit-message")
    cm.add_argument("--since", type=int, default=0, help="run id before this workflow started")
    args = ap.parse_args(argv)
    db_path = load_config(args.config).db_path

    if args.cmd == "last-run":
        print(last_run_id(db_path))
    elif args.cmd == "vacuum":
        before, after = vacuum(db_path)
        print(f"VACUUM {db_path.name}: {before / 1024:.0f} KB -> {after / 1024:.0f} KB")
    else:
        print(commit_message(db_path, args.since))
    return 0


if __name__ == "__main__":
    sys.exit(main())
