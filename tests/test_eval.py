"""The eval harness: folds, rerank ordering, leakage, caching, report. Fake LLM, no network."""

from dataclasses import replace

import pytest

import digest.eval as ev
from conftest import QUERY, RELEVANT_IDS, FakeLLM, batch_titles, reply
from digest.labels import LabelStore
from digest.retrieval.feedback import Example


def make_data(papers, n_copies=3):
    """30 papers: 3 copies of the 10 samples with new ids (4 relevant per copy)."""
    ps, rel = [], {}
    for c in range(n_copies):
        for p in papers:
            q = replace(p, id=f"{p.id}-c{c}")
            ps.append(q)
            rel[q.id] = int(p.id in RELEVANT_IDS)
    return ev.Dataset(ps, rel)


def test_two_folds_are_stratified_disjoint_and_deterministic(papers):
    data = make_data(papers)
    a, b = ev.two_folds(data)
    ids_a, ids_b = {p.id for p in a}, {p.id for p in b}
    assert not ids_a & ids_b and len(ids_a | ids_b) == 30
    assert sum(data.relevant[i] for i in ids_a) == 6 and sum(data.relevant[i] for i in ids_b) == 6
    assert [p.id for p in ev.two_folds(data)[0]] == [p.id for p in a]


def test_rerank_only_reorders_the_top_20(papers):
    data = make_data(papers)
    base = ev.rank_bm25(QUERY, data.papers)
    last = base[-1]  # outside the top 20
    scores = {pid: 1.0 for pid in base}
    scores[base[19]] = 10.0  # promote the 20th
    scores[last] = 10.0  # ...but this one isn't in the top 20, so it can't move
    ranked = ev.rank_rerank(QUERY, data.papers, scores)
    assert ranked[0] == base[19] and ranked[-1] == last
    assert ranked[20:] == base[20:]
    assert set(ranked[:20]) == set(base[:20])  # same top-20 set: R@20 can't change


def test_unscored_papers_keep_their_place_after_scored(papers):
    data = make_data(papers)
    base = ev.rank_bm25(QUERY, data.papers)
    ranked = ev.rank_rerank(QUERY, data.papers, {base[5]: 9.0})
    assert ranked[0] == base[5] and ranked[1:20] == [p for p in base[:20] if p != base[5]]


def test_auc():
    assert ev.auc({"a": 3, "b": 1, "c": 2}, {"a": 1, "b": 0, "c": 0}) == 1.0
    assert ev.auc({"a": 1, "b": 1}, {"a": 1, "b": 0}) == 0.5
    assert ev.auc({"a": 1, "b": 2, "c": 3}, {"a": 1, "b": 0, "c": 1}) == 0.5


def test_reported_half_never_trains_on_its_own_labels(papers, monkeypatch):
    data = make_data(papers)
    first = data.papers[0]
    data.reactions = [Example(first, 1, "reaction")]  # a reaction on a paper inside one half
    calls = []
    real = ev.rank_feedback

    def spy(query, ranked_papers, examples, weight):
        calls.append(({p.id for p in ranked_papers}, [e.paper.id for e in examples]))
        return real(query, ranked_papers, examples, weight)

    monkeypatch.setattr(ev, "rank_feedback", spy)
    result = ev.run_eval(data, QUERY, {})
    a, b = ({p.id for p in f} for f in result.folds)
    # 2 halves x (3 weights to choose + 1 report) = 8 calls. The last call per half is the report.
    assert len(calls) == 8
    for report_idx, (held_out, other) in ((3, (a, b)), (7, (b, a))):
        ranked, examples = calls[report_idx]
        assert ranked == held_out
        assert set(examples).isdisjoint(held_out), "held-out half leaked into its own training data"
        assert set(examples) <= other
        for choose_idx in range(report_idx - 3, report_idx):  # choosing ranks the training half itself
            assert calls[choose_idx][0] == other and set(calls[choose_idx][1]) <= other
    assert first.id in a  # the reaction sits in half A...
    assert first.id not in calls[3][1] and first.id in calls[7][1]  # ...so only half B's model may use it
    assert len(result.chosen_weights) == 2


def test_run_eval_rows_and_best_prompt(papers):
    data = make_data(papers)
    perfect = {p.id: 10.0 if data.relevant[p.id] else 1.0 for p in data.papers}
    inverted = {p.id: 1.0 if data.relevant[p.id] else 10.0 for p in data.papers}
    flat = {p.id: 9.0 for p in data.papers}
    result = ev.run_eval(data, QUERY, {"v1": inverted, "v2": perfect, "v3": flat})
    by = {r.method: r.mean for r in result.rows}
    assert set(by) == {"BM25 (current)", "TF-IDF", "BM25 + LLM rerank v1", "BM25 + LLM rerank v2",
                       "BM25 + LLM rerank v3", "BM25 + feedback (weight by 2-fold CV)",
                       "BM25 + feedback + LLM rerank v1", "BM25 + feedback + LLM rerank v2",
                       "BM25 + feedback + LLM rerank v3"}
    assert by["BM25 + feedback + LLM rerank v2"]["NDCG@10"] == pytest.approx(1.0)  # perfect LLM wins on any base
    assert by["BM25 + LLM rerank v2"]["NDCG@10"] == pytest.approx(1.0)
    assert by["BM25 + LLM rerank v1"]["NDCG@10"] < 0.5  # inverted scores push relevant papers down
    assert by["BM25 + LLM rerank v3"]["NDCG@10"] == pytest.approx(by["BM25 (current)"]["NDCG@10"])  # flat = no-op
    assert by["BM25 + LLM rerank v1"]["R@20"] == by["BM25 + LLM rerank v2"]["R@20"]  # rerank can't change R@20
    assert result.best_prompt() == "v2"

    report = ev.render_report(result, "fake/fake-1", "2026-11-26")
    assert "| BM25 + LLM rerank v2 | 1.00 |" in report
    assert "**Best rerank prompt by NDCG@10: v2.**" in report
    assert "30 hand-labelled papers (12 relevant, 18 not)" in report


def test_best_prompt_tie_prefers_the_earlier_version(papers):
    data = make_data(papers)
    perfect = {p.id: 10.0 if data.relevant[p.id] else 1.0 for p in data.papers}
    assert ev.run_eval(data, QUERY, {"v1": perfect, "v2": dict(perfect)}).best_prompt() == "v1"


def test_score_distribution_flags_bunching(papers):
    data = make_data(papers)
    flat = {p.id: 9.0 for p in data.papers}
    s = ev.score_distribution("v1", flat, data.relevant)
    assert s["share_top"] == 1.0 and s["distinct"] == 1 and s["hist"][9] == 30 and s["auc"] == 0.5
    v3 = ev.score_distribution("v3", {"a": 95, "b": 40}, {"a": 1, "b": 0})
    assert v3["top_cut"] == 90 and v3["share_top"] == 0.5 and v3["auc"] == 1.0


def test_llm_scores_are_cached_per_version(tmp_path, papers):
    def scorer(user):
        return reply({label: 9 if "RAG" in title or "Retrieval" in title else 2 for label, title in batch_titles(user)})

    with LabelStore(tmp_path / "labels.db", default_topic="test-topic") as store:
        llm = FakeLLM(default=scorer)
        first = ev.llm_scores_for("v1", papers, store, "fake/m", "topic", llm)
        assert len(first) == 10 and llm.calls == 2  # 10 papers / 5 per batch
        again = ev.llm_scores_for("v1", papers, store, "fake/m", "topic", FakeLLM())  # FakeLLM() fails if called
        assert again == first
        assert ev.llm_scores_for("v2", papers, store, "fake/m", "topic", None) == {}  # other version: not cached


def test_main_end_to_end(tmp_path, papers, capsys):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("topic:\n  name: RAG evaluation\n  keywords: [RAG evaluation]\n"
                   "database:\n  path: data/digest.db\n  labels_path: data/labels.db\n", encoding="utf-8")
    data = make_data(papers)
    with LabelStore(tmp_path / "data" / "labels.db", default_topic="rag-evaluation") as s:
        s.add_to_pool(data.papers)
        for pid, r in data.relevant.items():
            s.set_label(pid, bool(r))

    def scorer(user):
        return reply({label: 9 if "RAG" in title else 3 for label, title in batch_titles(user)})

    out = tmp_path / "report.md"
    code = ev.main(["--config", str(cfg), "--versions", "v1", "--out", str(out)],
                   make_llm=lambda provider, model: FakeLLM(default=scorer))
    assert code == 0
    text = out.read_text(encoding="utf-8")
    assert "BM25 + LLM rerank v1" in text and "## LLM score distribution" in text
    assert "# Ranking evaluation" in capsys.readouterr().out


def test_main_needs_labels(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("topic:\n  name: T\n  keywords: [k]\ndatabase:\n  path: d.db\n  labels_path: l.db\n",
                   encoding="utf-8")
    assert ev.main(["--config", str(cfg), "--no-llm"]) == 1


def test_random_baseline_matches_simulation(papers):
    import random
    import statistics

    from digest.metrics import ndcg_at_k, precision_at_k

    data = make_data(papers)
    exact = ev.random_baseline(data.papers, data.relevant)
    rng = random.Random(1)
    ids = [p.id for p in data.papers]
    sims = []
    for _ in range(20000):
        rng.shuffle(ids)
        sims.append((precision_at_k(ids, data.relevant, 5), ndcg_at_k(ids, data.relevant, 10)))
    assert exact["P@5"] == pytest.approx(12 / 30)
    assert exact["P@5"] == pytest.approx(statistics.fmean(s[0] for s in sims), abs=0.01)
    assert exact["NDCG@10"] == pytest.approx(statistics.fmean(s[1] for s in sims), abs=0.01)


def test_cohen_kappa():
    same = {"a": 1, "b": 0, "c": 1, "d": 0}
    assert ev.cohen_kappa(same, same) == (1.0, 4)
    flipped = {k: 1 - v for k, v in same.items()}
    assert ev.cohen_kappa(same, flipped)[0] == pytest.approx(-1.0)
    assert ev.cohen_kappa({"a": 1, "b": 0}, {"a": 1, "c": 0}) == (1.0, 1)
    k, n = ev.cohen_kappa(same, {})
    assert n == 0


def test_report_shows_random_row_and_kappa(papers):
    data = make_data(papers)
    data.previous_round = {p.id: 1 - r for p, r in ((p, data.relevant[p.id]) for p in data.papers[:10])}
    report = ev.render_report(ev.run_eval(data, QUERY, {}), "fake/m", "2026-11-26")
    assert "| *Random order (expected)* | 0.40 |" in report
    # 10 papers, 4 relevant, every label flipped: observed agreement 0, chance agreement
    # 0.4*0.6 + 0.6*0.4 = 0.48, so kappa = (0 - 0.48) / (1 - 0.48) = -0.92
    assert "Cohen's kappa -0.92 between this round and the previous labelling round on 10 papers" in report


# --- repeated splits ---
def test_seeded_splits_differ_but_stay_stratified_and_reproducible(papers):
    data = make_data(papers)
    canon = [p.id for p in ev.two_folds(data, 0)[0]]
    s1 = [p.id for p in ev.two_folds(data, 1)[0]]
    assert s1 != canon and s1 == [p.id for p in ev.two_folds(data, 1)[0]]
    for seed in range(5):
        a, b = ev.two_folds(data, seed)
        assert sum(data.relevant[p.id] for p in a) == 6 and len(a) + len(b) == 30


def test_rerank_on_a_custom_base_order(papers):
    data = make_data(papers)
    base = [p.id for p in reversed(data.papers)]
    ranked = ev.rank_rerank(QUERY, data.papers, {base[3]: 9.0}, base=base)
    assert ranked[0] == base[3] and ranked[20:] == base[20:]


def test_repeated_summary_and_wins(papers):
    data = make_data(papers)
    perfect = {p.id: 10.0 if data.relevant[p.id] else 1.0 for p in data.papers}
    inverted = {p.id: 1.0 if data.relevant[p.id] else 10.0 for p in data.papers}
    rep = ev.run_repeated(data, QUERY, {"v1": inverted, "v2": perfect}, repeats=4)
    assert len(rep.results) == 4 and rep.best_prompt() == "v2"
    mean, sd = rep.summary("BM25 + LLM rerank v2", "NDCG@10")
    assert mean == pytest.approx(1.0) and sd == pytest.approx(0.0)
    assert rep.wins("BM25 + LLM rerank v1", "BM25 (current)") == 0
    text = ev.render_repeated(rep)
    assert "## Main result: 4 repeated 2-fold splits" in text and "| BM25 + LLM rerank v1 |" in text
    assert "| *Random order (expected)* | 0.40 |" in text and "**Best rerank prompt by mean NDCG@10: v2.**" in text


def test_eval_reads_only_the_chosen_topics_labels(tmp_path, papers):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("topics:\n  - name: A\n    keywords: [k]\n  - name: B\n    keywords: [k]\n"
                   "database:\n  path: data/digest.db\n  labels_path: data/labels.db\n", encoding="utf-8")
    data = make_data(papers)
    with LabelStore(tmp_path / "data" / "labels.db", default_topic="a") as s:
        s.add_to_pool(data.papers, topic="b")
        for pid, r in data.relevant.items():
            s.set_label(pid, bool(r), topic="b")
    from digest.config import load_config
    assert len(ev.load_dataset(load_config(cfg), "a").papers) == 0
    assert len(ev.load_dataset(load_config(cfg), "b").papers) == 30
    out = tmp_path / "r.md"
    assert ev.main(["--config", str(cfg), "--topic", "b", "--no-llm", "--repeats", "2", "--out", str(out)]) == 0
    assert ev.main(["--config", str(cfg), "--topic", "a", "--no-llm"]) == 1  # a has no labels
