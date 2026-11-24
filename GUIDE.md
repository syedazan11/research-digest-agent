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
5. Open `data/candidates/<today>.json`. This file is what Day 3 (reranking) will read.

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

## Coming next

| Day | Build |
|---|---|
| 2 | Gemini Flash reranking: score top 20 from 1–10 with a reason (Pydantic-validated JSON) → top 5 |
| 3 | Download top 5 PDFs, extract sections with PyMuPDF, summarize each |
| 4 | Telegram bot: send the digest with 👍/👎 buttons |
| 5 | GitHub Actions: weekly digest run + feedback poller every few hours, commit the DB back |
| 6 | Feedback learning: boost papers similar to your 👍 ones |
| 7 | GitHub Pages archive, README results, demo video |
