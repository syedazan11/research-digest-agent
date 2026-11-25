# Research Digest Agent

An agent that tracks new arXiv papers on a topic you choose, ranks them with hybrid retrieval and LLM reranking, and sends a weekly digest to Telegram. It learns from your 👍/👎 ratings.

> **Status: Day 3 of 7.** Fetch → dedup → hybrid ranking → LLM rerank → PDF summaries → Markdown digest works. Summaries, Telegram, automation and feedback come next. See [GUIDE.md](GUIDE.md).

## Pipeline

```
arXiv fetch → SQLite dedup → Hybrid ranking (BM25 + Gemini embeddings) → top 20
   → LLM rerank (Groq gpt-oss-120b) → top 5 → PyMuPDF summaries → Telegram digest + GitHub Pages archive
Feedback poller (every few hours) → saves 👍/👎 into SQLite → improves next ranking
```

## Quickstart

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env            # then paste your Groq (and optionally Gemini) key into .env
python -m pytest                  # all offline, fake API clients
python -m digest.run              # real run
```

## Design decisions

| Decision | Choice | Why |
|---|---|---|
| Retrieval | BM25 + Gemini embeddings, pluggable `Retriever` interface | Keywords catch exact terms; embeddings catch paraphrases. New methods plug in without touching the pipeline. |
| Score fusion | Min-max normalize each retriever, weighted sum | BM25 and cosine scores live on different scales. |
| Embeddings | Gemini API, cached in SQLite | No PyTorch to install on CI; each paper is embedded once. |
| Reranking | LLM scores top 20 from 1-10 with a reason, Pydantic-validated, keeps top 5 | Similarity ≠ usefulness. Provider is swappable (Groq/Gemini) in `config.yaml`; any failure falls back to hybrid order. |
| Prompt injection | Abstracts escaped inside `<paper>` tags, labelled P1..P5, marked untrusted | Abstracts are third-party text; a paper must not be able to rank itself. |
| Summaries | Abstract + introduction + conclusion from the PDF (~3K tokens), five Pydantic-validated fields, cached per paper and model | Most of a paper's signal at a fraction of the tokens; any PDF failure falls back to the abstract with a recorded reason. |
| Storage | SQLite | One file, zero setup, commits back to the repo from GitHub Actions. |

## Results

_Coming after a few weekly runs: Precision@5 per retrieval method, cost per run, feedback impact._
