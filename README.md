# Research Digest Agent

[![Weekly digest](https://github.com/syedazan11/research-digest-agent/actions/workflows/weekly-digest.yml/badge.svg)](https://github.com/syedazan11/research-digest-agent/actions/workflows/weekly-digest.yml)
[![Tests](https://github.com/syedazan11/research-digest-agent/actions/workflows/tests.yml/badge.svg)](https://github.com/syedazan11/research-digest-agent/actions/workflows/tests.yml)

A weekly, multi-topic arXiv research digest that fetches papers, deduplicates per topic, ranks with BM25 and Rocchio feedback, reranks with a swappable LLM provider, summarizes PDF sections, and publishes one GitHub issue per topic. GitHub Actions runs the pipeline weekly, and issue reactions feed the next ranking cycle.

## Architecture

```text
Weekly GitHub Actions (for each configured topic)
arXiv fetch → topic-scoped dedup → BM25 + Rocchio feedback → LLM rerank
           → PDF section summaries → GitHub issue
                 ↑                                      │
                 └──────────── 👍/👎 reactions ─────────┘
```

## Results

On the 108-paper LLM-evaluation dataset, mean NDCG@10 across 20 repeated stratified two-fold splits was **0.43 random baseline → 0.57 with LLM reranking → 0.60 with feedback plus LLM reranking**. Labels are single-rater; agreement between rounds was low (**Cohen’s κ = 0.16**).

## Limitations

The evaluation covers one topic and a small, single-rater label set; the repeated splits overlap, and these labels now also train production feedback, so they are not a clean held-out test set. Recall is measured only within the keyword-selected candidate pool, so missed papers are not evaluated.

## Quickstart

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env            # then paste your Groq (and optionally Gemini) key into .env
python -m pytest                  # all offline, fake API clients
python -m digest.run --dry-run    # real run that writes and posts nothing
```

`--topic <slug>` runs one topic; every other command (`digest.label`, `digest.eval`, `digest.feedback`, `digest.deliver`, `digest.judge`) takes `--topic` too.

## Use it yourself in 5 minutes

All in the browser, no Python needed. You need a GitHub account and a free [Groq API key](https://console.groq.com/keys).

1. **Copy the repo:** click **Use this template → Create a new repository** (public, so Actions minutes are free).
2. **Pick your topics:** in your copy, open `config.yaml` → ✏️ edit, and replace the `topics:` list with yours. [`config.example.yaml`](config.example.yaml) has a commented two-topic example; each topic needs a `name`, `slug`, `description`, `keywords` and `categories` ([arXiv categories](https://arxiv.org/category_taxonomy)).
3. **Start with empty data:** **Actions → Reset data → Run workflow**, type `RESET`. Your copy came with my papers, runs and labels; this replaces them with empty databases (it refuses to run on the original repo).
4. **Add your key:** **Settings → Secrets and variables → Actions**:
   - **Secrets → New repository secret:** `GROQ_API_KEY` = your key
   - **Variables → New repository variable:** `DIGEST_ENABLED` = `true` (turns on the Monday schedule; without it, a fresh copy never posts on its own)
5. **Run it:** **Actions → Weekly digest → Run workflow**. A few minutes later each topic has a digest issue in your **Issues** tab. React 👍/👎 on papers; next Monday's run learns from it.

Keyword tip: a single broad word can flood a topic. Watch the run log for `Fetch hit max_results`. For *AI agents*, `agentic` alone matched 500+ papers a week; `agentic system` / `agentic workflow` matched 12-32.

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
| Feedback learning | Rocchio: TF-IDF similarity to 👍 papers minus 👎 papers, from reactions + hand labels, as a hybrid retriever | Cheap, explainable, needs no training run. A paper's own label is removed before it's scored, so it can't help itself. |
| Evaluation | Hand labels, P@5 / R@20 / NDCG@10, 20 repeated 2-fold splits, LLM scores cached | One split was too noisy to pick a winner (it chose v3; 20 splits chose v1). |
| Multiple topics | One run row, digest file, issue (`topic:<slug>` label) and feedback set per topic; dedup per topic; one LLM client with a per-run call budget | Topics stay independent (a paper can appear in two digests; a 👍 in one never affects another), and one topic's outage doesn't cost the others their digest. |
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

## Operational limitations

- **GitHub may silently stop the weekly schedule.** In a public repo, GitHub disables scheduled workflows after [60 days with no repository activity](https://docs.github.com/en/actions/using-workflows/disabling-and-enabling-a-workflow). The bot's own weekly commits most likely don't count: the [gh-action-keepalive](https://github.com/efrecon/gh-action-keepalive) project tried exactly that and reports it didn't work. Workflow runs don't count either. So after two months without a human push, the digest stops arriving. **What to do:** the missing Monday issue is the alarm. Re-enable the workflow in **Actions → Weekly digest → Enable workflow**, or push any commit. A keepalive step using a second token is possible but not set up.
- **Groq's free tier caps the number of topics.** 1000 requests/day and, the real limit, 8000 tokens/minute. Two topics take ~30 calls and ~4 minutes (mostly waiting for the per-minute token window); `llm.max_calls_per_run` (60) stops a bad week from burning the day's quota.
- **Scheduled runs can be late.** GitHub queues cron jobs and delays them at busy times, so "09:15" means "some time Monday morning".
- **arXiv from cloud runners.** arXiv can be slow or rate-limit requests from cloud IPs. A paper whose PDF can't be fetched is summarized from its abstract, and the digest says so.
- **Rerank scores bunch up.** With prompt v1, top papers often share a 9/10 and BM25 breaks the ties. Day 6 tested stricter prompts: they spread the scores out but didn't rank better, so v1 stays (see Results).

## Evaluation details

Ranking evaluation for the **LLM evaluation** topic, 2026-11-26 ([full report](data/evals/2026-11-26.md), `python -m digest.eval --topic llm-evaluation`). The *AI agents* topic has no labels yet.

**Dataset:** 108 papers from a one-off 30-day arXiv fetch with the topic keywords, hand-labelled by me against a written rubric: *y = the paper's main contribution is about evaluating LLMs; n = it only uses an LLM or benchmark as a tool, or evaluation is a side detail.* 46 relevant, 62 not.

**Protocol:** 20 repeated, stratified 2-fold splits. Every method ranks one half at a time against the topic query; the table shows mean ± standard deviation over the 20 splits. The feedback weight is chosen on one half and reported on the other, so a half never trains on its own labels.

| Method | P@5 | NDCG@10 | beats BM25 (same split) |
|---|---|---|---|
| Random order (expected) | 0.43 | 0.426 | |
| BM25 | 0.45 ± 0.10 | 0.432 ± 0.047 | |
| TF-IDF | 0.40 ± 0.07 | 0.437 ± 0.028 | 9/20 |
| BM25 + LLM rerank, prompt v1 (original) | 0.59 ± 0.07 | 0.568 ± 0.029 | 20/20 |
| BM25 + LLM rerank, prompt v2 (strict rubric) | 0.62 ± 0.07 | 0.525 ± 0.034 | 20/20 |
| BM25 + LLM rerank, prompt v3 (strict, 1-100) | 0.53 ± 0.06 | 0.553 ± 0.043 | 20/20 |
| BM25 + feedback learning | 0.47 ± 0.07 | 0.486 ± 0.053 | 17/20 |
| **BM25 + feedback → LLM rerank v1 (production)** | **0.64 ± 0.08** | **0.600 ± 0.037** | 20/20 |

What it says:
- **BM25 alone barely beats random** within the keyword-filtered pool: every candidate already matches the keywords, so keyword overlap can't tell them apart.
- **The LLM rerank is the big win** (NDCG@10 0.43 → 0.57, better on every split).
- **The stricter prompts fixed score bunching but not ranking.** v3 uses 35 distinct scores instead of v1's 10, yet v1 ranks at least as well (it beats v3 on 14/20 splits, within one sd). So v1 stays the default.
- **Feedback learning helps modestly:** +0.055 over BM25 (17/20 splits) and +0.032 on top of the rerank (15/20). It's on in production at a 20% weight share, the value the inner cross-validation chose most often.

**Summary quality** ([report](data/evals/2026-11-25-summaries.md), `python -m digest.judge`): 6 summaries judged field by field against their source text by a different model family (Qwen3.8 27B judging gpt-oss-120b). 29/30 fields supported, 1 partly (a "why it matters" claimed a safety benefit the paper doesn't make; confirmed by hand), 0 unsupported. All 6 within the length rules, checked by code.

**Caveats, honestly:**
- **Small and noisy.** 108 labels from one person on one topic. A first labelling round, done too fast, scored at chance for every method and contradicted my own 👍/👎. I relabelled against a written rubric; the two rounds agree only slightly (Cohen's kappa 0.16, partly because the rubric was stricter). The ranking of methods is more trustworthy than the exact numbers.
- **The 20 splits overlap**, so "20/20" is 20 views of the same 108 papers, not 20 independent experiments.
- **These labels can no longer be a clean test set.** Production feedback learning now trains on them, so a future version evaluated on them would be graded on its own training data. **A fresh, held-out batch of labels is future work** before comparing new versions.
- **Recall is within the candidate pool only:** the pool comes from the same keyword search as production, so papers the keywords miss are invisible to this eval.
- LLM scores are cached per paper but were produced in batches of 5, so a paper's score can depend on its batch-mates.
