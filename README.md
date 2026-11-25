# Research Digest Agent

[![Weekly digest](https://github.com/syedazan11/research-digest-agent/actions/workflows/weekly-digest.yml/badge.svg)](https://github.com/syedazan11/research-digest-agent/actions/workflows/weekly-digest.yml)
[![Tests](https://github.com/syedazan11/research-digest-agent/actions/workflows/tests.yml/badge.svg)](https://github.com/syedazan11/research-digest-agent/actions/workflows/tests.yml)

An agent that tracks new arXiv papers on a topic you choose, ranks them with hybrid retrieval and LLM reranking, and posts a weekly digest as a GitHub issue. It learns from your 👍/👎 reactions.

> **Status: Day 5 of 7.** The full pipeline runs unattended every Monday on GitHub Actions: fetch → dedup → hybrid ranking → LLM rerank → PDF summaries → GitHub issue → 👍/👎 feedback. Feedback learning and the archive come next. See [GUIDE.md](GUIDE.md).

## Pipeline

```
GitHub Actions, Mondays 09:15 PKT:
  python -m digest.feedback → owner's 👍/👎 reactions from last week → SQLite
  python -m digest.run:
    arXiv fetch → SQLite dedup → Hybrid ranking (BM25 + Gemini embeddings) → top 20
      → LLM rerank (Groq gpt-oss-120b) → top 5 → PyMuPDF summaries
      → GitHub issue (1 comment per paper) + Markdown digest
  commit data/digest.db back to the repo
```

## Quickstart

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env            # then paste your Groq (and optionally Gemini) key into .env
python -m pytest                  # all offline, fake API clients
python -m digest.run --dry-run    # real run that writes and posts nothing
```

To run it weekly on your own fork: add a `GROQ_API_KEY` repository secret, then use **Actions → Weekly digest → Run workflow** once. See GUIDE.md, Day 5.

## Design decisions

| Decision | Choice | Why |
|---|---|---|
| Retrieval | BM25 + Gemini embeddings, pluggable `Retriever` interface | Keywords catch exact terms; embeddings catch paraphrases. New methods plug in without touching the pipeline. |
| Score fusion | Min-max normalize each retriever, weighted sum | BM25 and cosine scores live on different scales. |
| Embeddings | Gemini API, cached in SQLite | No PyTorch to install on CI; each paper is embedded once. |
| Reranking | LLM scores top 20 from 1-10 with a reason, Pydantic-validated, keeps top 5 | Similarity ≠ usefulness. Provider is swappable (Groq/Gemini) in `config.yaml`; any failure falls back to hybrid order. |
| Prompt injection | Abstracts escaped inside `<paper>` tags, labelled P1..P5, marked untrusted | Abstracts are third-party text; a paper must not be able to rank itself. |
| Summaries | Abstract + introduction + conclusion from the PDF (~3K tokens), five Pydantic-validated fields, cached per paper and model | Most of a paper's signal at a fraction of the tokens; any PDF failure falls back to the abstract with a recorded reason. |
| Delivery | GitHub issue per run, one comment per paper; reactions = feedback | No bot server, works where Telegram is blocked. Resumable: DB records + hidden markers mean a crash never double-posts. |
| Automation | GitHub Actions, weekly cron + manual trigger, built-in `GITHUB_TOKEN` (contents + issues write only) | Free, no server. `--strict` makes any fallback fail the job (GitHub emails you), but whatever succeeded is still committed. |
| Storage | SQLite file committed to the repo by the workflow | One file, zero setup, versioned with the code. Trade-offs below. |

### Why the SQLite database lives in git

| Option | For | Against |
|---|---|---|
| **DB committed to `main`** (chosen) | Zero infrastructure, free; every week's state is in git history; saved by the same job that made it | Binary: no diffs, and conflicts can't be merged (CI owns the file and wins). The repo grows by about one DB-sized blob per week (VACUUM keeps it small: ~140 KB now). The data is public. |
| Separate `data` branch | Keeps `main`'s history free of bot commits | A slightly more complex workflow |
| Actions cache / artifacts | No commits at all | Caches are evicted after 7 days unused; artifacts expire after 90 days. The data would quietly disappear. |
| Hosted DB (Turso, Supabase, Neon) | A real database: concurrent writers, queries from anywhere | Another account, secret and network dependency for a few hundred KB of data |
| Git LFS | Keeps the repo history small | Free tier has storage and bandwidth limits; extra tooling for contributors |

At ~140 KB plus a few KB a week, committing to `main` stays cheap for years. If it ever grows past a few MB, a `data` branch is the next step.

## Known limitations

- **GitHub may silently stop the weekly schedule.** In a public repo, GitHub disables scheduled workflows after [60 days with no repository activity](https://docs.github.com/en/actions/using-workflows/disabling-and-enabling-a-workflow). The bot's own weekly commits most likely don't count: the [gh-action-keepalive](https://github.com/efrecon/gh-action-keepalive) project tried exactly that and reports it didn't work. Workflow runs don't count either. So after two months without a human push, the digest stops arriving. **What to do:** the missing Monday issue is the alarm. Re-enable the workflow in **Actions → Weekly digest → Enable workflow**, or push any commit. A keepalive step using a second token is possible but not set up.
- **Scheduled runs can be late.** GitHub queues cron jobs and delays them at busy times, so "09:15" means "some time Monday morning".
- **arXiv from cloud runners.** arXiv can be slow or rate-limit requests from cloud IPs. A paper whose PDF can't be fetched is summarized from its abstract, and the digest says so.
- **Rerank scores bunch up.** The top 5 often all score 9/10, so BM25 breaks the ties. This is the first thing the Day 6 evaluation looks at.

## Results

_Coming after a few weekly runs: Precision@5 per retrieval method, cost per run, feedback impact._
