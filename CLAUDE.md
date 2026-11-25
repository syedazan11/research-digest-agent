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
- Day 3 done: PDF sections + summaries + Markdown digest (digest/pdf.py, summarize.py, render.py, summaries table)
- Day 4 next: Telegram bot with feedback buttons (see GUIDE.md)

## Eval backlog (Day 6)
1. LLM rerank scores bunch up. Real run #3 (2026-11-24, 19 papers): the top 5 all scored 9, so the
   hybrid (BM25) score decided their order, and HighwayVerse (a road-inspection benchmark) got an 8.
   Don't change the prompt blind: first measure Precision@5 from the `rankings` table + feedback,
   then try a stricter rubric or a 1-100 scale and compare against that baseline.
