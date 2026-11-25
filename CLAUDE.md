# Research Digest Agent

Weekly arXiv digest: fetch -> dedup -> hybrid rank -> LLM rerank -> summarize -> GitHub issue -> learn from reactions.
I'm a student building this for my portfolio. Explain design decisions briefly so I can defend them in interviews.

## Run / test
- Windows, PowerShell, venv in .venv
- Tests: python -m pytest (must stay offline: use fake clients, never real API calls)
- Pipeline: python -m digest.run   (offline: --sample tests/fixtures/sample_papers.json --dry-run)

## Rules
- Never read, print or commit .env
- CI owns data/digest.db (committed by the weekly workflow). Never commit local DB changes; `git pull`
  first, use --dry-run for experiments. Workflow env must never use real secrets in tests.yml
- Tests must never reach real GitHub: use tests/fake_github.py (httpx.MockTransport); test_run stubs make_github
- Keep pluggable designs: retrievers follow digest/retrieval/base.py; LLM providers are swappable via config.yaml
- A failing API must never crash the run: log a clear warning and fall back
- Add/update tests for every change; all tests must pass before you finish

## Status
- Day 1 done: arXiv fetch, SQLite dedup, BM25 ranking (Gemini embeddings off: Google `AQ.`-prefixed keys return 401)
- Day 2 done: LLM reranking with Groq (digest/llm.py, digest/rerank.py, rankings table)
- Day 3 done: PDF sections + summaries + Markdown digest (digest/pdf.py, summarize.py, render.py, summaries table)
- Day 4 done: GitHub Issues delivery + reaction feedback (digest/github.py, deliver.py, feedback.py;
  digest_issues + deliveries tables). Telegram dropped: blocked in Pakistan.
- Day 5 done: GitHub Actions. weekly-digest.yml (Mon 04:15 UTC: digest.feedback -> digest.run --strict ->
  commit data/digest.db as github-actions[bot]) and tests.yml (no secrets). No separate feedback poller:
  reactions persist, so they're read once before each weekly run.
- Day 6 next: feedback learning + evals (see Eval backlog)

## Eval backlog (Day 6)
1. LLM rerank scores bunch up. Real run #3 (2026-11-24, 19 papers): the top 5 all scored 9, so the
   hybrid (BM25) score decided their order, and HighwayVerse (a road-inspection benchmark) got an 8.
   Don't change the prompt blind: first measure Precision@5 from the `rankings` table + feedback,
   then try a stricter rubric or a 1-100 scale and compare against that baseline.
