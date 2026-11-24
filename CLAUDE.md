# Research Digest Agent

Weekly arXiv digest: fetch -> dedup -> hybrid rank -> LLM rerank -> summarize -> Telegram -> learn from feedback.
I'm a student building this for my portfolio. Explain design decisions briefly so I can defend them in interviews.

## Run / test
- Windows, PowerShell, venv in .venv
- Tests: python -m pytest (must stay offline: use fake clients, never real API calls)
- Pipeline: python -m digest.run   (offline: --sample tests/fixtures/sample_papers.json --dry-run)

## Rules
- Never read, print or commit .env
- Keep pluggable designs: retrievers follow digest/retrieval/base.py; LLM providers are swappable via config.yaml
- A failing API must never crash the run: log a clear warning and fall back
- Add/update tests for every change; all tests must pass before you finish

## Status
- Day 1 done: arXiv fetch, SQLite dedup, BM25 ranking (Gemini embeddings off: Google `AQ.`-prefixed keys return 401)
- Day 2 done: LLM reranking with Groq (digest/llm.py, digest/rerank.py, rankings table)
- Day 3 next: PDFs + PyMuPDF summaries (see GUIDE.md)
