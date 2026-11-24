# Build Guide

Your step-by-step guide for building the Research Digest Agent. Each day ends with a checklist. Don't move on until it's all ticked.

---

## Day 1: Fetch → Dedup → Hybrid Ranking

**Goal:** by tonight, one command pulls this week's papers from arXiv, skips ones you've seen, and prints the 20 most relevant, scored by keywords (BM25) and meaning (Gemini embeddings).

### What's in the folder

```
research-digest-agent/
├── config.yaml              ← YOUR topic, keywords, weights. Edit this first.
├── .env.example             ← copy to .env, put your Gemini key in it
├── requirements.txt
├── pyproject.toml           ← tells pytest where the code is
├── README.md                ← the public face of the project (recruiters read this)
├── GUIDE.md                 ← this file
├── data/                    ← SQLite DB + saved candidates appear here
├── digest/
│   ├── run.py               ← the pipeline: python -m digest.run
│   ├── config.py            ← loads config.yaml
│   ├── models.py            ← the Paper dataclass
│   ├── fetch.py             ← stage 1: arXiv search
│   ├── db.py                ← SQLite: seen papers, embedding cache, feedback, run log
│   ├── text.py              ← tokenizer for BM25
│   └── retrieval/
│       ├── base.py          ← the Retriever interface (2 members)
│       ├── bm25.py          ← keyword scoring
│       ├── gemini_embed.py  ← semantic scoring + cache + retries
│       └── hybrid.py        ← normalizes and combines scores
└── tests/
    ├── fixtures/sample_papers.json   ← 10 SYNTHETIC papers for offline testing
    ├── test_fetch.py
    ├── test_db.py
    └── test_retrieval.py
```

### Step 1: Set up (≈20 min)

1. **Python 3.10 or newer.** Check with `python --version`.
2. Open the folder in VS Code, then open a terminal (`Ctrl + ` `).
3. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```
   You'll see `(.venv)` at the start of the prompt. In VS Code, also pick this interpreter: `Ctrl+Shift+P` → *Python: Select Interpreter* → `.venv`.
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Get a Gemini API key (free) at https://aistudio.google.com/apikey, then:
   ```bash
   copy .env.example .env
   ```
   Open `.env` and replace `your-key-here` with your key.
6. Start git **before** your first commit so `.env` is never committed:
   ```bash
   git init
   git add .
   git status        # .env must NOT be in the list
   git commit -m "Day 1: project skeleton"
   ```
   Then create an empty GitHub repo and push.

### Step 2: Run the tests (≈5 min)

```bash
python -m pytest
```
Expect **15 passed**. These run fully offline. The Gemini test uses a fake client, so it costs nothing.

### Step 3: Offline run on sample data (≈5 min)

```bash
python -m digest.run --sample tests/fixtures/sample_papers.json --no-embeddings --dry-run
```
You'll see the 4 RAG papers on top and the 6 unrelated ones at 0.00.

**Look closely at #4** ("Grounded Answer Generation..."). It's on-topic but scores only 0.08 because it never uses the words "RAG" or "retrieval-augmented". That's BM25's weakness, and it's why we add embeddings. Note this down: it's a perfect README / interview example.

### Step 4: Pick your topic (≈15 min)

Edit `config.yaml`:
- `name` + `description`: write the description like you'd explain the topic to a friend. It's the ranking query.
- `keywords`: 3–6 phrases. arXiv must match at least one in the title or abstract. Too narrow → 0 papers; too broad → 200 off-topic ones.
- `categories`: browse https://arxiv.org/category_taxonomy

### Step 5: Real runs (≈20 min)

1. **BM25 only** (checks arXiv works):
   ```bash
   python -m digest.run --no-embeddings --dry-run
   ```
2. **Full hybrid run:**
   ```bash
   python -m digest.run
   ```
   Each candidate shows `bm25 0.xx  embedding 0.xx`, so you can see which retriever liked it.
3. **Run it again immediately.** It should say `0 new (N already seen)`. That's dedup working.
4. Check the embedding cache saved money:
   ```bash
   python -m digest.run --include-seen
   ```
   The log line `Gemini embedding API calls this run` should say **0**: every paper and the query come from the cache.
5. Open `data/candidates/<today>.json`. This file is what Day 2 (reranking) builds on.

### Step 6: Look inside the database (≈10 min)

Install the VS Code extension **"SQLite Viewer"**, then click `data/digest.db`. Check the `papers`, `embeddings` and `runs` tables. The `runs` table is your future proof that the agent ran unattended.

### Step 7: Commit

```bash
git add .
git commit -m "Day 1: arXiv fetch, SQLite dedup, BM25 + Gemini hybrid ranking"
git push
```

### Day 1 checklist

- [ ] `python -m pytest` → 15 passed
- [ ] Offline sample run works; you can explain why paper #4 scores low
- [ ] `config.yaml` set to YOUR topic
- [ ] Real run prints ranked papers with both `bm25` and `embedding` scores
- [ ] Second run shows `0 new`
- [ ] `.env` is not on GitHub
- [ ] Pushed to GitHub

### Understand it (interviewers will ask)

Try answering these out loud without looking:

1. **Why min-max normalize before combining?** BM25 scores might run 0–15 while cosine similarity runs 0.4–0.8. Without normalizing, BM25 would dominate whatever the weights say.
2. **Why both BM25 and embeddings?** BM25 is exact and explainable but misses paraphrases (paper #4). Embeddings catch meaning but can rank vaguely related papers too high. Together they cover each other.
3. **Why `RETRIEVAL_QUERY` vs `RETRIEVAL_DOCUMENT` task types?** Gemini optimizes the embedding for its role: short questions vs long documents.
4. **Why normalize vectors ourselves?** At 768 dimensions, `gemini-embedding-001` doesn't return unit-length vectors, so a raw dot product would not be cosine similarity.
5. **Why strip the version from arXiv ids?** So `2411.01234v2` isn't treated as a new paper after you've already seen v1.
6. **Why is the client injectable in `GeminiEmbeddingRetriever`?** So tests can pass a fake and run offline, free and fast.

### Troubleshooting

| Problem | Fix |
|---|---|
| `ImportError: DLL load failed` (numpy etc.) | Windows Smart App Control is blocking Python DLLs. Turn it off, or use GitHub Codespaces. |
| `Could not reach arXiv` | Check internet. arXiv's API is sometimes slow; just retry in a minute. |
| `0 fetched` | Keywords too narrow or `days_back` too small. Try `days_back: 14` or broader keywords. |
| `GEMINI_API_KEY not set` warning | `.env` missing, or still says `your-key-here`. Must be in the project folder. |
| Gemini `429` in logs | Free-tier rate limit. The code retries automatically (5s, 10s, 20s). |
| `ModuleNotFoundError: digest` | Run from the project folder with `python -m digest.run`, not `python digest/run.py`. |

---

## Day 2: LLM Reranking

**Goal:** an LLM reads the hybrid top 20, scores each paper 1–10 for how useful it is (with a one-line reason), and keeps the best 5. If the LLM is down, the run still finishes with the hybrid top 5.

### What's new

```
digest/
├── llm.py       ← swappable LLM providers: GroqLLM, GeminiLLM, make_llm()
└── rerank.py    ← batching, prompt, Pydantic validation, retry, fallback
config.yaml      ← new rerank: section
data/digest.db   ← new rankings table (one row per candidate per run)
```

### Step 1: Groq key (≈5 min)

1. Get a free key at https://console.groq.com/keys.
2. Add it to `.env` (see `.env.example`): `GROQ_API_KEY=...`
3. `pip install -r requirements.txt` (adds `groq` and `pydantic`).

### Step 2: Tests (≈2 min)

```bash
python -m pytest
```
All offline. The LLM tests use a `FakeLLM` with scripted replies (good JSON, garbage, exceptions), so they cost nothing and give the same result every time.

### Step 3: Offline run

```bash
python -m digest.run --sample tests/fixtures/sample_papers.json --no-embeddings --dry-run --no-rerank
```
You'll see the hybrid top 10, then a `Rerank (rerank off)` table. Drop `--no-rerank` to send the 10 sample papers to Groq (2 calls).

### Step 4: Real run (≈5 min)

Save the Day 1 output first (`data/candidates/<date>.json` → copy it somewhere), then:

```bash
python -m digest.run --include-seen
```

The rerank table looks like this (illustrative; `*` = selected for the digest):

```
 hybrid -> final  llm   title
* #7   ->   1    9.0  Judge Models ...
                       Directly studies LLM-as-a-judge reliability.
  #2   ->   -    4.0  Clinical Agents ...
```

Compare with Day 1: which papers moved, and do the reasons convince you?

### Step 5: Look in the database

`rankings` has one row per candidate per run: `hybrid_rank`, `hybrid_score`, `llm_score`, `reason`, `final_rank`, `selected`. Once you have 👍/👎 feedback (Day 4+), this is what you compute **Precision@5** from: hybrid top 5 vs LLM top 5.

### Day 2 checklist

- [ ] `python -m pytest` passes
- [ ] Real run shows the rerank table with LLM scores and reasons
- [ ] `--no-rerank` and a missing `GROQ_API_KEY` both still finish with a hybrid top 5
- [ ] `rankings` table has rows for the run
- [ ] Compared Day 1 vs Day 2 top 5 and noted one example for the README
- [ ] Committed and pushed

### Understand it (interviewers will ask)

1. **Why rerank with an LLM at all?** BM25 and embeddings measure *similarity* to the topic text. An LLM can judge *usefulness*: a clinical-agents paper can mention "evaluation" a lot and still not be about evaluating LLMs.
2. **Why only the top 20, and why batches of 5?** Cost and latency: cheap retrieval narrows 200 → 20, the expensive model only sees 20. Batches of 5 keep the prompt short (the model compares papers side by side but doesn't lose track), and one bad batch only affects 5 papers.
3. **Why P1..P5 labels instead of arXiv ids?** Short, can't be mistyped into another real id, trivial to map back. Unknown ids in the reply are simply ignored.
4. **Why Pydantic?** The LLM's output is untrusted input. Pydantic checks the shape and the 1–10 range in one line, and its error message is exactly what we send back on the retry.
5. **What stops prompt injection?** Abstracts are third-party text. They go inside `<paper>` tags with `<`/`>` escaped (so an abstract can't close its tag), the system prompt says never to follow instructions inside papers, and the output is a fixed schema — the worst an injected abstract can do is lie about its own score.
6. **What happens to papers the LLM skips?** They go after all scored papers, in hybrid order. An unchecked paper shouldn't outrank one the LLM actually judged.
7. **Ties?** Broken by hybrid score — the cheaper signal is still information.
8. **What if Groq is down?** 429 → wait for `retry-after` and retry. Bad key → stop immediately (retrying can't fix it). Anything else → retry the batch once, then skip it. If no batch succeeds, the run uses the hybrid top 5 and says so in the log and the `runs.notes` column.
9. **What if a retriever fails?** `HybridRanker` logs a warning, skips it, and renormalizes the weights over the retrievers that worked. `runs.retrievers` records which ones actually ran.
10. **Why temperature 0?** Ranking should be repeatable: same papers, same scores, so runs are comparable.
11. **How do you switch to Gemini?** `rerank.provider: gemini` and a model name in `config.yaml`. Both providers implement the same one-method `LLMClient` protocol.

### Troubleshooting

| Problem | Fix |
|---|---|
| `GROQ_API_KEY not set: LLM reranking disabled` | Add the key to `.env` in the project folder. |
| `Groq rejected the API key (401)` | Key typo or revoked. Make a new one. The run still finishes with the hybrid top 5. |
| `Groq 429 rate limit, waiting Ns` | Free-tier limit. It waits and retries automatically. |
| `Batch N/M failed after retry` | The model returned invalid JSON twice. Those 5 papers keep hybrid order. |
| `LLM reranking failed completely` | Every batch failed. Check the warnings above it. |

---

## Coming next

| Day | Build |
|---|---|
| 3 | Download top 5 PDFs, extract sections with PyMuPDF, summarize each |
| 4 | Telegram bot: send the digest with 👍/👎 buttons |
| 5 | GitHub Actions: weekly digest run + feedback poller every few hours, commit the DB back |
| 6 | Feedback learning: boost papers similar to your 👍 ones |
| 7 | GitHub Pages archive, README results, demo video |
