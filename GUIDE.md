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

## Coming next

| Day | Build |
|---|---|
| 6 | Feedback learning: boost papers similar to your 👍 ones |
| 7 | GitHub Pages archive, README results, demo video |
