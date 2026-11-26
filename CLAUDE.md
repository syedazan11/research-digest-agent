# Research Digest Agent

Weekly arXiv digest: fetch -> dedup -> hybrid rank -> LLM rerank -> summarize -> GitHub issue -> learn from reactions.
I'm a student building this for my portfolio. Explain design decisions briefly so I can defend them in interviews.

## Run / test
- Windows, PowerShell, venv in .venv
- Tests: python -m pytest (must stay offline: use fake clients, never real API calls)
- Pipeline: python -m digest.run   (offline: --sample tests/fixtures/sample_papers.json --dry-run)

## Rules
- Never read, print or commit .env
- data/labels.db is yours (you commit it); CI only opens it read-only. Never write to it from the pipeline
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
- Day 6 done: hand labels (digest/label.py, data/labels.db), eval harness (digest/eval.py, metrics.py),
  Rocchio feedback retriever (retrieval/feedback.py, tfidf.py), rerank prompt versions, summary judge (judge.py)
- Day 7 next: GitHub Pages archive, README results polish, demo video

## Eval results and backlog
- Day 6 eval (data/evals/2026-11-26.md, 108 labels, 20 repeated 2-fold splits): LLM rerank v1 is the big win
  (NDCG@10 0.43 -> 0.57); stricter prompts v2/v3 reduced score bunching but didn't rank better, so v1 stays.
  Feedback learning on at a 20% share (feedback: 0.1 next to bm25: 0.4; test_config keeps them in step).
- Labels in data/labels.db now train production feedback, so they are NOT a clean test set for future
  versions. Before comparing a new ranker, label a fresh held-out batch (new pool, new round).
- Label quality matters more than the code: round 1 (fast, no rubric) was at chance for every method.
  Keep topic.label_rubric, check AUC vs random and kappa vs the previous round before trusting a result.
