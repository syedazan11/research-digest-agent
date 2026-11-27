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
5. Open `data/candidates/<date>_run<N>.json` (one file per logged run; `--dry-run` writes none). This file is what Day 2 (reranking) builds on.

### Step 6: Look inside the database (≈10 min)

Install the VS Code extension **"SQLite Viewer"**, then click `data/digest.db`. Check the `papers`, `embeddings` and `runs` tables. The `runs` table is your future proof that the agent ran unattended.

### Step 7: Commit

```bash
git add .
git commit -m "Day 1: arXiv fetch, SQLite dedup, BM25 + Gemini hybrid ranking"
git push
```

### Day 1 checklist

- [ ] `python -m pytest` → 285 passed
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

Each logged run gets its own `data/candidates/<date>_run<N>.json`, so your Day 1 file stays. Then:

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

## Day 3: PDFs → Summaries → Markdown Digest

**Goal:** the top 5 papers each get a five-part summary (problem, method, key result, why it matters, limitations) built from the actual paper, and the run writes a clean Markdown digest that Days 4 (GitHub Issues) and 7 (GitHub Pages) reuse.

### What's new

```
digest/
├── pdf.py         ← download from export.arxiv.org (3s apart), extract abstract + intro + conclusion
├── summarize.py   ← five Pydantic-validated fields, cached in SQLite
├── prompting.py   ← shared injection guard + validate-and-retry loop (rerank uses it too)
└── render.py      ← data/digests/<date>_run<N>.md
config.yaml        ← new summarize: section
data/digest.db     ← new summaries table
```

### Step 1: Install (≈2 min)

```bash
pip install -r requirements.txt     # adds pymupdf and httpx
python -m pytest
```
All offline: the tests build tiny PDFs with PyMuPDF, use a fake HTTP function instead of arXiv, and a `FakeLLM` instead of Groq.

### Step 2: Real run (≈3 min)

```bash
python -m digest.run --include-seen
```
Watch the log: one `PDF text for <id>: abstract+introduction+conclusion, ~2200 tokens` line per paper, then:

```
summaries: 5/5 (5 from PDF, 0 abstract-only, 0 cached)
LLM calls this run: 17 (rerank 6, summarize 11)
Timing: fetch 2.0s | rank 0.0s | rerank 22.3s | summarize 101.9s | total 126.2s
```

Most of the summarize time is waiting on Groq's free-tier limit (8K tokens/minute): you'll see `Groq 429 rate limit, waiting 15.0s (retry-after)`. That's expected. The call count includes those retried requests.

Open `data/digests/<date>_run<N>.md` (VS Code: `Ctrl+Shift+V` for the preview).

### Step 3: Run it again

```bash
python -m digest.run --include-seen
```
Now: `summarize 0` calls, `PDF downloads this run: 0`, and the summarize stage takes ~0s. Every summary came from the cache.

### Step 4: Look in the database

`summaries` has one row per paper per model: the JSON summary, `source` (`pdf` or `abstract`) and, for abstract-only ones, `fallback_reason`:

| fallback_reason | Meaning |
|---|---|
| `download_failed` | arXiv unreachable, timeout, HTTP error, or no PDF link |
| `not_pdf` | got bytes, but not a readable PDF (e.g. an HTML error page) |
| `no_text` | PDF has (almost) no text: scanned images |
| `no_sections` | text found, but no Introduction or Conclusion heading |

### Day 3 checklist

- [ ] `python -m pytest` passes
- [ ] Real run writes `data/digests/<date>_run<N>.md` with 5 summaries
- [ ] Second run makes 0 summarize calls and 0 PDF downloads
- [ ] `--no-summarize` still writes a digest (with abstracts)
- [ ] You read the 5 summaries against the papers: anything invented?
- [ ] Committed and pushed

### Understand it (interviewers will ask)

1. **Why not send the whole PDF?** Groq's free tier is ~8K tokens a minute and a paper is 10–30K. The introduction states the problem and claims, the conclusion says what worked, and with the abstract that's ~3K tokens: most of the signal at a fraction of the cost.
2. **Why the arXiv metadata abstract instead of the PDF's?** It's already clean text. In the PDF, the abstract is tangled with the title, authors and affiliations.
3. **How do you find sections?** A heading heuristic on extracted lines: optionally numbered `Introduction` / `Conclusion(s)` headings, ending at the next top-level numbered heading or References/Appendix. A real paper (2611.28940) broke it twice: figure text `0 Favors A` and a running header `2 Preprint` looked like headings. Fixes: section numbers start at 1, and short lines that repeat on many pages are dropped as headers/footers. That's now a regression test.
4. **Why prefer Conclusion over Discussion?** Discussion often re-argues results; Conclusion is the authors' own summary of what holds. Discussion/Summary is used only when there's no Conclusion.
5. **Why count tokens as characters ÷ 4?** It's a budget, not a bill. Within ~10% is fine, and it avoids a tokenizer dependency that wouldn't match Groq's model anyway.
6. **Why a 3-second throttle?** arXiv's API etiquette asks for at most one request every 3 seconds. The User-Agent names the project and links the repo so arXiv can see who's calling.
7. **What if the PDF fails?** The summary is built from the abstract and marked "based on the abstract only (reason)" in the digest. The reason is stored, so you can count how often each failure happens.
8. **Why is an abstract-only summary final?** "Never summarize the same paper twice" is a simple, testable rule. The cost: a brief arXiv outage leaves that paper with a thinner summary forever. A weekly digest can live with that.
9. **Why cache by (paper, provider/model)?** Re-runs cost nothing, and switching models gives fresh summaries instead of mixing two models' output.
10. **How do you stop the model inventing limitations?** The prompt says to use only the provided text and to write "Not stated in the provided text." when there are none. Pydantic also caps each field at 500 characters, so a rambling reply is retried.
11. **What stops a malicious PDF?** Same guard as rerank, from shared code (`prompting.py`): the text is escaped inside a `<paper>` tag, the system prompt says never to follow instructions in it, and the output is a fixed schema. In the Markdown, every paper/LLM string is escaped, so injected text can't become a link or an image.
12. **Why one LLM client for rerank and summarize?** One key, one config line, and one call counter, so the log can say `17 (rerank 6, summarize 11)`.
13. **Why Markdown?** One format both later stages can use: GitHub Issues render it, GitHub Pages publishes it.

### Troubleshooting

| Problem | Fix |
|---|---|
| `PDF fallback for <id> (download_failed ...)` | arXiv slow or down. That paper uses its abstract; the run continues. |
| `Groq 429 rate limit, waiting Ns` | Free-tier token limit. Expected during summaries; it waits and continues. |
| `No summary for <id> after retry` | Model gave invalid JSON twice. The digest shows the abstract; the next run tries again (failures aren't cached). |
| Digest shows `?` in the terminal | Windows console can't display some characters. The `.md` file is fine (UTF-8). |

---

## Day 4: Deliver as a GitHub Issue → 👍/👎 Feedback

**Goal:** each run opens one GitHub issue in this repo, posts one comment per paper, and your 👍/👎 reactions on those comments flow back into the `feedback` table. (Originally planned as a Telegram bot; Telegram is blocked in Pakistan, and GitHub Issues needs no bot server at all.)

### What's new

```
digest/
├── github.py     ← small GitHub REST client: rate limits, clear token errors
├── deliver.py    ← issue + comments, resumable; also `python -m digest.deliver`
└── feedback.py   ← `python -m digest.feedback`: reactions -> feedback table
config.yaml       ← new deliver: section
data/digest.db    ← new digest_issues + deliveries tables; feedback table migrated
```

### Step 1: Token (≈5 min)

1. GitHub → Settings → Developer settings → **Fine-grained tokens** → Generate new token.
2. Repository access: **Only select repositories** → this repo.
3. Permissions → Repository → **Issues: Read and write** (Metadata: read is added automatically).
4. Add to `.env` (see `.env.example`):
   ```
   GITHUB_TOKEN=github_pat_...
   GITHUB_REPO=your-username/research-digest-agent
   ```

### Step 2: Tests, then a preview

```bash
python -m pytest
python -m digest.deliver --dry-run
```
The tests run the real GitHub client against an in-memory fake GitHub (`tests/fake_github.py`). `--dry-run` prints the issue and every comment for the newest undelivered run, and calls nothing.

### Step 3: Deliver for real

```bash
python -m digest.deliver          # post the newest undelivered run
python -m digest.run              # or: the full pipeline, which delivers at the end
```
Open the repo's **Issues** tab: one issue labelled `digest`, five comments. Run `python -m digest.deliver` again: it says `0 comments posted, 5 already there`.

### Step 4: Feedback

React 👍 or 👎 on a few paper comments (from your own account), then:
```bash
python -m digest.feedback
```
→ `Feedback: 5 comments checked, 2 added, ...`. Take a reaction back and run it again: that row is removed. Check the `feedback` table in SQLite Viewer.

### Day 4 checklist

- [ ] `python -m pytest` passes
- [ ] `--dry-run` preview looks right
- [ ] A real digest issue with 5 comments in the repo
- [ ] Re-running `python -m digest.deliver` posts nothing new
- [ ] Your 👍/👎 show up in `feedback` after `python -m digest.feedback`
- [ ] `.env` still not committed; committed and pushed

### Understand it (interviewers will ask)

1. **Why GitHub Issues instead of Telegram/email?** Free, no bot server, works in Pakistan, notifications by email and the mobile app, and reactions are a built-in 👍/👎 UI with an API. The digest also lives next to the code, which is nice for a portfolio.
2. **How is delivery idempotent?** Two layers. The DB records the issue the moment it's created and each comment the moment it's posted, so a re-run skips them. But a crash can land after GitHub saved a post and before the DB write. So every issue and comment carries a hidden marker (`<!-- digest:run=4 paper=... -->`), and a resume first lists the issue's comments and adopts any it finds. The DB is the fast path; GitHub is the source of truth.
3. **Why rebuild the digest from the DB to deliver it?** So any run can be delivered or resumed later (`python -m digest.deliver --run 4`) without re-fetching or re-paying for LLM calls. The `.md` file uses the same DB-backed renderer, so the file and the issue can't drift apart.
4. **Why escape `@` and `#`?** Paper and LLM text is untrusted. `@someone` in an abstract would ping a real person; `#12` would link a random issue. I checked GitHub's own renderer (`POST /markdown`): a backslash does **not** stop the autolinker. An invisible word joiner (U+2060) after the `@`/`#` does, and the text still reads the same.
5. **Why only the owner's reactions?** The repo is public. Anyone could react, and strangers shouldn't steer the ranking.
6. **"Latest state wins"?** The feedback table mirrors GitHub: newest 👍/👎 counts, a removed reaction deletes the row. Polling twice changes nothing. A paper whose comment was deleted keeps its old rating (we can't see GitHub's state, so we don't guess).
7. **Rate limits?** Primary limit: wait until `x-ratelimit-reset`. Secondary limit: wait for `retry-after`, or 60s if absent (GitHub's guidance). Waits are capped; past the cap it stops with a clear error and a resume command, instead of hanging a CI job for an hour. Comments are posted ~1s apart, which GitHub asks for on content-creating requests.
8. **What if the token is bad?** 401/403 become a `GitHubAuthError` with a fix-it message ("needs Issues: Read and write"). The pipeline logs it and still exits 0 with the `.md` digest saved; the standalone `deliver`/`feedback` commands exit 1 so CI shows the failure. The token never appears in logs.
9. **Why a fine-grained token?** Least privilege: one repo, one permission. If it leaks, the damage is "someone can edit issues on this repo".
10. **Why `httpx.MockTransport` instead of mocking the client?** The tests run the real request loop, headers, pagination and rate-limit code against a fake server. Mocking the client would test the mock.
11. **Why does `--dry-run` save the run to the DB?** It saves to the in-memory copy. The preview is then built by exactly the code a real delivery uses, and nothing touches disk.

### Troubleshooting

| Problem | Fix |
|---|---|
| `GITHUB_TOKEN not set: skipping GitHub delivery` | Add `GITHUB_TOKEN` and `GITHUB_REPO` to `.env`. |
| `GITHUB_TOKEN is invalid or expired (401)` | Token expired (fine-grained tokens have an expiry). Make a new one. |
| `lacks permission ... needs Issues: Read and write` | Edit the token's permissions, or it was made for another repo. |
| `Repo ... not found, or GITHUB_TOKEN can't access it` | Check `GITHUB_REPO=owner/name` and the token's repository access. |
| `GitHub delivery stopped ... Resume with: python -m digest.deliver --run N` | Rate limit or outage. Run that command later; nothing is posted twice. |
| Reactions not counted | Only the repo owner's count. Add others to `deliver.feedback_users`. |

---

## Day 5: Run Unattended on GitHub Actions

**Goal:** every Monday morning, with nobody at the keyboard, GitHub Actions collects last week's reactions, runs the whole pipeline, posts the digest issue, and commits the updated database back to the repo. A second workflow runs the tests on every push and PR.

### What's new

```
.github/workflows/
├── weekly-digest.yml   ← Mondays 04:15 UTC (09:15 PKT) + "Run workflow" button
└── tests.yml           ← pytest on every push / PR, no secrets
digest/maintenance.py   ← last-run, vacuum, commit-message (the workflow's DB chores)
data/digest.db          ← now tracked in git; CI commits it back after each run
.gitattributes          ← LF line endings for workflows, *.db is binary
```
`python -m digest.run --strict` is new too: if any stage fell back (no LLM, rerank failed, no summaries, GitHub delivery failed) it exits 1 *after* saving everything, so the job fails visibly.

### Design change: no separate feedback poller

The original plan had a second workflow polling reactions every few hours. Not needed: **reactions persist on GitHub**, so reading them once, right before each weekly run, gives exactly the same data. One workflow instead of two, ~5 API requests a week instead of ~200, and no race between the poller and the weekly run committing the DB at the same time.

### Step 1: Add the secret (≈3 min)

Repo → **Settings → Secrets and variables → Actions → New repository secret**: `GROQ_API_KEY` = your Groq key. That's the only secret. GitHub delivery uses the built-in `GITHUB_TOKEN`, which the workflow limits to `contents: write` + `issues: write`.

### Step 2: Merge, then run it once by hand

The schedule and the "Run workflow" button only work once the workflow file is on `main`. After merging: **Actions → Weekly digest → Run workflow**. Watch the steps; the job summary shows the digest. Then check:
- a new digest issue (labelled `digest`), and the previous one closed
- a commit on `main` by `github-actions[bot]`: `digest: run N (5 papers, issue #M)`
- the **Tests** workflow ran on your merge, but *not* on the bot's commit (pushes made with `GITHUB_TOKEN` never trigger other workflows: no loops)

### Step 3: Pull before you work locally

CI now owns `data/digest.db`. Before running anything locally: `git pull`. For experiments use `--dry-run` (writes nothing). If you changed the DB by accident: `git checkout data/digest.db`. Two copies of a binary file can't be merged; if both change, CI's wins.

### Day 5 checklist

- [ ] `GROQ_API_KEY` secret added
- [ ] Tests workflow green on the PR
- [ ] Manual "Run workflow" posts a digest issue and a bot commit
- [ ] Both README badges show passing
- [ ] Next Monday: a digest arrived without you doing anything

### Understand it (interviewers will ask)

1. **Why GitHub Actions?** Free for public repos, no server to run or patch, secrets management built in, and the logs of every run are public proof the agent runs unattended.
2. **Why 04:15 UTC and not 04:00?** Scheduled runs are queued, and the top of the hour is GitHub's busiest time, when they get delayed. Pakistan is UTC+5 with no daylight saving, so 04:15 UTC is always 09:15 PKT.
3. **Why `permissions: contents: write, issues: write` only?** Least privilege. If a dependency were compromised during the run, the token couldn't touch PRs, packages, deployments or settings.
4. **What does the concurrency group do?** A manual run while the scheduled one is going would post a second digest and race to push the DB. `cancel-in-progress: false` makes the second run *wait*: cancelling a run halfway through posting comments is worse than waiting.
5. **How do failures work?** The pipeline and commit steps use `if: !cancelled()`, so they still run after an earlier step failed. Whatever succeeded (feedback, run log, deliveries) is committed; the job is still red, and GitHub emails you. `--strict` turns quiet fallbacks into that red job, without losing the work.
6. **Why check for DB changes before VACUUM?** I simulated the commit step against throwaway git repos: with an unchanged DB it still committed, because VACUUM rewrites the SQLite header every time. Checking `git diff` first fixed it.
7. **Two people pushing at once?** The commit step does `git pull --rebase -X theirs` before pushing, with 3 retries. An unrelated push rebases cleanly; a conflicting `digest.db` keeps CI's copy (`theirs` = the commit being replayed, during a rebase). Both cases were simulated.
8. **Why keep SQLite in git?** Zero infrastructure and full history for a ~140 KB file. Costs: binary (no merges), repo growth, public data. Alternatives and their costs are in the README table: cache/artifacts expire, a hosted DB adds an account and a network dependency.
9. **Why doesn't CI use real secrets?** Tests must be reproducible and free, and PRs from forks must never see secrets. The tests use fakes for Groq, GitHub and arXiv; the workflow also blanks every key, so a missing stub fails instead of calling a real API.
10. **Does the bot's commit trigger the tests workflow?** No. GitHub never starts workflows from pushes made with `GITHUB_TOKEN`, specifically to prevent infinite loops.
11. **Can GitHub stop the schedule?** Yes: public repos lose scheduled workflows after 60 days without repository activity, and the bot's commits probably don't count. Mitigation: the missing Monday issue is the alarm, and re-enabling is one click. See README → Known limitations.

### Troubleshooting

| Problem | Fix |
|---|---|
| No "Run workflow" button | The workflow file isn't on `main` yet. Merge first. |
| `GROQ_API_KEY not set` / `Degraded run: LLM rerank fell back` | Add the repository secret (Step 1). |
| `GitHub delivery failed (... 403 ...)` | Settings → Actions → General → Workflow permissions: allow the workflow to request write access, or an org policy is blocking it. |
| `Push failed (attempt 3)` | Branch protection on `main` blocks the bot. Allow `github-actions[bot]`, or commit to a `data` branch. |
| No digest this Monday | Check Actions: if "This scheduled workflow is disabled", click **Enable workflow** (60-day rule). If it's just late, wait: cron runs can be delayed. |
| Local `git pull` conflicts on `data/digest.db` | `git checkout --theirs data/digest.db` (keep CI's), or discard yours with `git checkout origin/main -- data/digest.db`. |

---

## Day 6: Evaluation and Feedback Learning

**Goal:** stop guessing. Build a hand-labelled test set, measure every ranking method on it, fix the default prompt and the feedback weight from the numbers, and learn from your 👍/👎 and labels.

### What's new

```
digest/
├── label.py / labels.py      ← python -m digest.label: y/n labelling CLI, data/labels.db
├── metrics.py                ← Precision@k, Recall@k, NDCG@k (hand-computed tests)
├── eval.py                   ← python -m digest.eval: 20 repeated 2-fold splits, report in data/evals/
├── judge.py                  ← python -m digest.judge: LLM-as-judge for summary faithfulness
└── retrieval/
    ├── tfidf.py              ← numpy TF-IDF baseline
    └── feedback.py           ← Rocchio FeedbackRetriever (reactions + labels, no self-leakage)
data/labels.db                ← label pool, labels, archived rounds, cached LLM scores (you commit it)
```
Rerank now has prompt versions (`rerank.prompt_version`: v1 original, v2 strict rubric, v3 strict on 1-100), and production adds the feedback retriever with `ranking.weights.feedback`.

### Step 1: Build the label pool and label it (≈30 min)

```bash
python -m digest.label --fetch --days 30   # one-off: ~110 papers into data/labels.db only
python -m digest.label                     # y / n / s(kip) / u(ndo) / q(uit), saved per key
```
The rubric from `config.yaml` (`topic.label_rubric`) is on every screen. **Write it before you start, and take 20-30 s per paper.** Papers come in a fixed shuffled order with no score shown, so the current ranker can't anchor you.

**What happened the first time (a real lesson):** 110 labels in 8.5 minutes, no written rubric. Every method, including the LLM, scored exactly at chance (AUC 0.49-0.51), and the labels contradicted 3 of my 4 👍/👎. Not a bug: I checked the label-to-paper mapping and an off-by-one shift. The labels were noise. `python -m digest.label --relabel` archived them as round 1 and I started again with a rubric. The two rounds agree only slightly (Cohen's kappa 0.16).

### Step 2: Evaluate (≈1 min; the first run spends ~70 Groq calls)

```bash
python -m digest.eval        # report: data/evals/<date>.md
```
LLM scores are cached in `labels.db` per (paper, model, prompt version), so re-runs cost nothing.

### Step 3: Read it, then decide

The report has the random-order bar to beat, every method's mean ± sd over 20 splits, how often each beats BM25 on the same split, the LLM score distribution per prompt (bunching, AUC), and label consistency vs the previous round. Change a default only when the evidence is clearer than the noise.

### Step 4: Summary quality (optional)

```bash
python -m digest.judge       # report: data/evals/<date>-summaries.md
```

### Day 6 checklist

- [ ] Labelled with a written rubric, at a sane pace
- [ ] `python -m digest.eval` report committed with `data/labels.db`
- [ ] Defaults in `config.yaml` justified by the report (prompt version, feedback weight)
- [ ] README Results section with the caveats

### Understand it (interviewers will ask)

1. **Why these three metrics?** P@5 is what the digest delivers (5 papers). R@20 asks whether the 20 papers sent to the LLM contained the good ones; reranking can't fix what retrieval missed. NDCG@10 rewards putting relevant papers *near the top*, so it's the one used to pick prompts.
2. **Why is R@20 identical for every rerank row?** Reranking only reorders BM25's top 20, so the set is the same. A metric that can't move is worth saying out loud, not hiding.
3. **Why a random baseline?** The pool comes from the keyword search, so 43% of papers are relevant before any ranking. BM25's P@5 of 0.45 sounds fine until you see random gets 0.43.
4. **Why 20 repeated splits instead of one?** On the canonical split v3 ranked first; across 20 splits v1 did. With ~54 papers per half, one split is mostly luck. Mean ± sd plus "wins on the same split" separates signal from noise. Caveat: the splits overlap, so 20/20 isn't 20 independent experiments.
5. **How is leakage prevented?** Two layers. Inside the retriever: when a paper that is itself a training example gets scored, its own vector is removed from the profile first (tested against retraining without it). In the eval: to report half A, the feedback model trains only on half B's labels (and 👍/👎 on papers outside A), and its weight is chosen on B. Tested with a spy on every call.
6. **Why Rocchio and not a classifier?** ~110 examples. Rocchio has no training run, two interpretable knobs (β = 0.75 for 👍, γ = 0.25 for 👎: positives say more), handles cold start (no ratings → no effect), and plugs into the existing Retriever interface.
7. **Why keep labels and reactions separate (`source`)?** They're different signals: a label is a deliberate judgement against a rubric; a reaction is judgement of a delivered digest. If both exist, the reaction wins. And it keeps the door open to weighting them differently.
8. **The stricter prompt fixed bunching. Why not ship it?** Because the goal is ranking, not a nice histogram. v3 spread scores over 35 values instead of 10, but v1 still ranked at least as well (14/20 wins over v3, within one sd). The backlog item is closed with evidence, not with the prompt that looked better.
9. **How did you pick the feedback weight?** Nested: the inner CV chose among 0.2 / 0.35 / 0.5 on the training half; 0.2 won most often (21 of 40 halves) and is the most conservative. Production normalizes weights over the active retrievers, so `feedback: 0.1` next to `bm25: 0.4` is the same 20% share; a test keeps them in step.
10. **What's the biggest weakness of this eval?** The labels. One person, ~110 papers, kappa 0.16 between rounds. And because production now trains on them, they can't be a clean test set for the next version: a fresh held-out batch is needed first.
11. **Why judge summaries with a different model?** Models tend to rate their own output kindly. gpt-oss writes, Qwen judges. Length rules are checked by code, because counting sentences shouldn't need an LLM. The one flagged field was confirmed by hand.
12. **What did labelling badly teach you?** That a confident pipeline on bad labels produces confident nonsense. The sanity checks that caught it (chance-level AUC for *every* method, disagreement with my own reactions, labelling speed) are now part of the report.

### Troubleshooting

| Problem | Fix |
|---|---|
| `Need at least 10 labelled papers with both classes` | Label more: `python -m digest.label`. |
| Every method at AUC ≈ 0.5 | Suspect the labels before the code: re-read the rubric, check pace, compare with your 👍/👎. `--relabel` starts a new round (the old one is archived). |
| `vN: only k/n papers scored, left out` | A Groq outage mid-run. Re-run `python -m digest.eval`: only the missing papers are scored. |
| Judge: `model ... does not exist` | List your Groq models; pass one with `--judge-model`. |

---

## Day 7: Multiple Topics, and "Use It Yourself"

**Goal:** follow several research topics with one agent (each topic its own digest, dedup, feedback and labels), without losing any existing data, and make the repo something anyone can copy and run in five minutes.

### What's new

```
config.yaml            ← topics: [...] (the old single topic: still works)
config.example.yaml    ← commented two-topic starting point for your own copy
digest/
├── db.py              ← schema 2: topic_papers (per-topic "seen"), topic on runs/issues/feedback/summaries
├── labels.py          ← schema 2: pool, labels, rounds, LLM cache per topic
├── run.py             ← loops over topics; --topic <slug>; one topic failing doesn't stop the others
├── llm.py             ← BudgetedLLM (llm.max_calls_per_run), Groq quota from x-ratelimit headers
├── identity.py        ← User-Agent from project.url / GITHUB_REPOSITORY, never another copy's
└── maintenance.py     ← reset --yes (refuses on the original repo)
.github/workflows/reset-data.yml   ← "Reset data" button for a fresh copy (type RESET)
```

### Step 1: Add a topic (≈10 min)

Add an entry to `topics:` in `config.yaml`: `name`, `slug` (stable, never rename it), `description`, `keywords`, `categories`, `label_rubric`, and optionally `top_n`. **Keep `llm-evaluation` first**: data from before multi-topic support was migrated to `topics[0]`.

Then check the keywords against a real week before trusting them:
```bash
python -m digest.run --topic ai-agents --dry-run --no-summarize
```
Look for `N fetched` and any `Fetch hit max_results` warning. The first draft for *AI agents* hit the 200 cap: `agentic` alone matched 500+ papers a week, and `multi-agent` was half drones and multi-agent RL. `agentic system` / `agentic workflow` / `agentic AI` / `multi-agent LLM` brought it to 165 papers, 88% about LLMs or agentic systems.

### Step 2: The migration (automatic)

The first run on the new code migrates both databases once:
- `data/digest.db`: CI migrates it on the next Monday run and commits it (git history is the backup).
- `data/labels.db`: already migrated and committed in this branch, because CI only ever opens it read-only.

Every existing row (23 papers, 5 runs, 6 summaries, 4 ratings, 108 labels, 333 cached scores) moved to `llm-evaluation`; row counts were checked table by table, and the Day 6 eval reproduces exactly from the migrated labels.

### Step 3: Run both topics

```bash
python -m digest.run --dry-run       # both topics, nothing written or posted
```
Each topic ends with a line like `Topic ai-agents: ok, 5 papers, LLM calls 18 (rerank 8, summarize 10), 141s`, then `LLM calls this run: 30 of a 60 budget | Groq quota: requests 983/1000 left today, tokens 68/8000 left this minute`.

### Step 4 (optional): make it a template

Settings → General → tick **Template repository**. The README's "Use it yourself in 5 minutes" section is the whole setup for someone else.

### Day 7 checklist

- [ ] `python -m pytest` passes
- [ ] A dry run of both topics, with no `Fetch hit max_results` warning
- [ ] Next Monday: one issue per topic, each labelled `topic:<slug>`, and the bot's commit lists both runs
- [ ] (Template) "Reset data" refuses on your repo and works on a test copy

### Understand it (interviewers will ask)

1. **Why is dedup per topic?** "Seen" means "already shown in *this* digest". A paper about evaluating agents belongs in both an LLM-evaluation and an AI-agents digest; global dedup would let whichever topic runs first steal it.
2. **Why summaries per (paper, model, topic)?** `why_it_matters` is written relative to the topic. The same paper matters for different reasons to an evaluation researcher and an agents researcher. The PDF text is shared (downloaded once per run); the summary isn't.
3. **How did you migrate without losing data?** A schema version (`PRAGMA user_version`), one transaction for the whole migration, a `.bak` copy first, and every old row moved to `topics[0]`. SQLite can't change a primary key, so `summaries` and every `labels.db` table were rebuilt. Tested on the exact Day-6 schema (copied from `main`), including a simulated failure halfway (rolled back cleanly, migrated fine next time), and on a copy of the real database: all 7 tables kept their row counts.
4. **What if a tool opens the old database without saying which topic?** It refuses with a clear error instead of guessing. Silently tagging your data as a "default" topic would be a bug you'd find months later.
5. **Why does CI migrate `digest.db` but not `labels.db`?** CI owns `digest.db` and commits it; `labels.db` is yours and CI opens it read-only. So `labels.db` is migrated locally and committed in the PR, and if CI ever met an old file it migrates an in-memory copy instead of writing.
6. **What happens if one topic crashes?** The exception is logged with the topic's slug, the loop moves on, and the other topics still get their digest. `--strict` then fails the job at the end so GitHub emails you; a run where *every* topic failed exits 1 even without it.
7. **How do you stay inside Groq's free tier?** Measured, not guessed: Groq's `x-ratelimit-*` headers say 1000 requests/day and 8000 tokens/minute. Two topics use ~30 calls (3% of the day); the per-minute token window is what slows runs down, and the existing retry-after handling waits for it. A shared `BudgetedLLM` caps calls per run (60) so a bad week can't burn the day; past it, the remaining work falls back and the run is marked degraded.
8. **Why one arXiv client and one PDF fetcher for all topics?** arXiv asks for one request every 3 seconds. Separate clients per topic would each respect it and together break it.
9. **How do you keep a template copy from doing damage?** Three guards. Reset refuses without `--yes` and always on the original repo (checked from `GITHUB_REPOSITORY` and the git remote). A fresh copy's Monday schedule does nothing until its owner sets `DIGEST_ENABLED=true`, so it never posts digests about *my* topics. And the User-Agent comes from the copy's own repo, so it never claims to be mine.
10. **Why a browser-only reset workflow?** The person copying the template may not have Python installed. "Type RESET" makes it deliberate; sharing the weekly run's concurrency group means it can never run during a digest.
11. **How does feedback stay per topic?** A reaction is tagged with the topic of the issue it was left on (`deliveries` → `runs.topic`), and each topic's `FeedbackRetriever` trains only on that topic's reactions and labels. The dry run showed it: llm-evaluation trained on 108 examples, ai-agents on 0.

### Troubleshooting

| Problem | Fix |
|---|---|
| `This database is from before multi-topic support...` | A tool opened `digest.db` without a topic. Use `Database.for_config(cfg)`. |
| `Fetch hit max_results=300` | A keyword is too broad. Measure each keyword alone (see Step 1) and replace it with a phrase. |
| `LLM budget of 60 calls per run used up` | Too many topics for one run, or a lot of retries. Raise `llm.max_calls_per_run` or drop a topic. |
| `Topic X failed` but the job is green | That's the non-strict mode. CI uses `--strict`, so there it's red. Check the logged traceback. |
| A copy's Monday run does nothing | Set the repository variable `DIGEST_ENABLED=true`. |
| `Reset refused: ... original project` | You're on syedazan11/research-digest-agent. Reset is only for copies. |

---

## Coming next

| Day | Build |
|---|---|
| 7 | GitHub Pages archive, README results, demo video |
