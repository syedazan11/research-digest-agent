"""Small DB chores for the GitHub Actions workflows, kept in Python so they're testable.

    python -m digest.maintenance last-run                  # newest run id (0 if none)
    python -m digest.maintenance vacuum                    # compact the DB before committing it
    python -m digest.maintenance commit-message --since 4  # "digest: llm-evaluation run 5 (5 papers, issue #6)"
    python -m digest.maintenance reset --yes               # fresh copy: empty databases (backups first)
"""

from __future__ import annotations

import argparse
import re
import shutil
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
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
    """What the bot's commit says: every topic's run logged since `since_run`, e.g.
    "digest: llm-evaluation run 8 (5 papers, issue #9); ai-agents run 9 (5 papers, issue #10)",
    or that only feedback changed."""
    with sqlite3.connect(db_path) as conn:
        has_topic = "topic" in {r[1] for r in conn.execute("PRAGMA table_info(runs)")}
        runs = conn.execute(f"SELECT id, {'topic' if has_topic else 'NULL'} FROM runs WHERE id > ? ORDER BY id",
                            (since_run,)).fetchall()
        if not runs:
            n = conn.execute("SELECT COUNT(*) FROM feedback WHERE source = 'github'").fetchone()[0]
            return f"digest: feedback only ({n} ratings)"
        entries = []
        for run_id, topic in runs:
            n_papers = conn.execute("SELECT COUNT(*) FROM rankings WHERE run_id = ? AND selected = 1",
                                    (run_id,)).fetchone()[0]
            issue = conn.execute("SELECT issue_number FROM digest_issues WHERE run_id = ?", (run_id,)).fetchone()
            parts = [f"{n_papers} paper{'s' if n_papers != 1 else ''}"]
            if issue is not None:
                parts.append(f"issue #{issue[0]}")
            entries.append(f"{topic + ' ' if topic else ''}run {run_id} ({', '.join(parts)})")
    return "digest: " + "; ".join(entries)


# --- reset, for a fresh copy made from the template ---
PROTECTED_REPO = "syedazan11/research-digest-agent"  # the original: its data is never reset


class ResetRefused(RuntimeError):
    pass


def current_repo(cwd: Path | None = None) -> str:
    """owner/name of this checkout: GITHUB_REPOSITORY in Actions, else the git remote 'origin'."""
    import os

    repo = os.getenv("GITHUB_REPOSITORY", "").strip()
    if repo:
        return repo
    try:
        url = subprocess.run(["git", "config", "--get", "remote.origin.url"], cwd=cwd, capture_output=True,
                             text=True, timeout=10).stdout.strip()
    except Exception:
        return ""
    m = re.search(r"github\.com[:/]([^/]+/[^/]+?)(?:\.git)?/?$", url)
    return m.group(1) if m else ""


def reset_data(cfg, yes: bool, repo: str) -> list[str]:
    """Back up, then empty digest.db and labels.db (fresh schema-2 files) and remove eval reports, which
    describe someone else's labels. Refuses without yes=True, and always on the original repo."""
    from digest.db import Database
    from digest.labels import LabelStore

    if repo.lower() == PROTECTED_REPO:
        raise ResetRefused(f"refusing to reset data on {PROTECTED_REPO}: this is the original project, "
                           "not a copy. Resetting is for repos created from the template.")
    if not yes:
        raise ResetRefused("reset deletes this copy's history (seen papers, runs, feedback, labels). "
                           "Run again with --yes to confirm; backups are written first.")
    done = []
    stamp = f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
    for path in (cfg.db_path, cfg.labels_path):
        if path is not None and path.exists():
            backup = path.with_name(f"{path.name}.bak-reset-{stamp}")
            shutil.copy2(path, backup)
            path.unlink()
            done.append(f"backed up {path.name} to {backup.name}")
    with Database(cfg.db_path, default_topic=cfg.topic.slug):
        pass  # creates an empty schema-2 database
    if cfg.labels_path is not None:
        with LabelStore(cfg.labels_path, default_topic=cfg.topic.slug):
            pass
    done.append(f"created empty {cfg.db_path.name} and {cfg.labels_path.name if cfg.labels_path else ''}".strip())
    data_dir = cfg.db_path.parent
    for sub_dir in ("evals", "candidates", "digests"):
        d = data_dir / sub_dir
        if d.exists():
            shutil.rmtree(d)
            done.append(f"removed {sub_dir}/")
    return done


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="DB chores for the weekly workflow")
    ap.add_argument("--config", default="config.yaml")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("last-run")
    sub.add_parser("vacuum")
    cm = sub.add_parser("commit-message")
    cm.add_argument("--since", type=int, default=0, help="run id before this workflow started")
    rs = sub.add_parser("reset", help="empty this copy's databases (refuses on the original repo)")
    rs.add_argument("--yes", action="store_true", help="confirm: backups are written first")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    db_path = cfg.db_path

    if args.cmd == "reset":
        try:
            for line in reset_data(cfg, args.yes, current_repo(Path(args.config).resolve().parent)):
                print(line)
        except ResetRefused as e:
            print(f"Reset refused: {e}", file=sys.stderr)
            return 1
        print("Reset done. Next: edit config.yaml's topics, add the GROQ_API_KEY secret, run the workflow.")
        return 0

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
