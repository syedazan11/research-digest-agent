"""Label storage and the labelling loop (keys scripted, no terminal)."""

from digest.label import label_loop
from digest.labels import LabelStore, shuffle_key


def keys(*seq):
    it = iter(seq)
    return lambda: next(it)


def store_with(tmp_path, papers):
    s = LabelStore(tmp_path / "labels.db")
    s.add_to_pool(papers)
    return s


def test_pool_is_shuffled_deterministically_and_dedups(tmp_path, papers):
    s = store_with(tmp_path, papers)
    assert s.add_to_pool(papers) == 0  # nothing new
    order = [p.id for p in s.pool()]
    assert order == sorted((p.id for p in papers), key=shuffle_key)
    assert order != [p.id for p in papers]  # not file order: no ranking hints
    s.close()


def test_labels_and_resume(tmp_path, papers):
    s = store_with(tmp_path, papers)
    first, second, third = (p.id for p in s.unlabelled()[:3])
    counts = label_loop(s, keys("y", "n", "q"), out=lambda _: None)
    assert counts == {"relevant": 1, "not_relevant": 1, "skipped": 0, "undone": 0}
    assert s.labels() == {first: 1, second: 0}
    s.close()

    s = LabelStore(tmp_path / "labels.db")  # come back later
    assert s.unlabelled()[0].id == third  # resumes at the first unlabelled paper
    assert s.stats() == {"pool": 10, "relevant": 1, "not_relevant": 1, "unlabelled": 8}
    s.close()


def test_skip_and_undo(tmp_path, papers):
    s = store_with(tmp_path, papers)
    ids = [p.id for p in s.unlabelled()]
    shown = []
    label_loop(s, keys("y", "s", "n", "u", "y", "x", "q"), out=shown.append)
    # y on 0, skip 1, n on 2, undo (back to 2), y on 2, unknown key, quit
    assert s.labels() == {ids[0]: 1, ids[2]: 1}
    assert any("undone" in o for o in shown) and any("[y] relevant" in o for o in shown)
    s.close()


def test_undo_with_nothing_to_undo(tmp_path, papers):
    s = store_with(tmp_path, papers)
    shown = []
    label_loop(s, keys("u", "q"), out=shown.append)
    assert "(nothing to undo)" in shown and s.labels() == {}
    s.close()


def test_labelled_papers_and_relabel(tmp_path, papers):
    s = store_with(tmp_path, papers)
    s.set_label(papers[0].id, True)
    s.set_label(papers[0].id, False)  # change of mind
    assert [(p.id, r) for p, r in s.labelled_papers()] == [(papers[0].id, 0)]
    s.close()


def test_read_only_never_creates_or_changes_the_file(tmp_path, papers):
    missing = tmp_path / "nope" / "labels.db"
    with LabelStore(missing, read_only=True) as ro:
        assert ro.labels() == {} and ro.pool() == []
    assert not missing.parent.exists()

    path = tmp_path / "labels.db"
    with LabelStore(path) as s:
        s.add_to_pool(papers)
        s.set_label(papers[0].id, True)
    before = path.read_bytes()
    with LabelStore(path, read_only=True) as ro:
        assert ro.labels() == {papers[0].id: 1}
    assert path.read_bytes() == before


def test_llm_score_cache(tmp_path):
    with LabelStore(tmp_path / "labels.db") as s:
        s.put_llm_score("p1", "groq/m", "v1", 9, "good")
        s.put_llm_score("p1", "groq/m", "v2", 6, "ok")
        assert s.get_llm_scores("groq/m", "v1") == {"p1": (9.0, "good")}
        assert s.get_llm_scores("groq/m", "v3") == {}


def test_empty_pool_message(tmp_path):
    shown = []
    with LabelStore(tmp_path / "labels.db") as s:
        label_loop(s, keys("q"), out=shown.append)
    assert "Fetch a pool first" in shown[0]


# --- relabelling ---
def test_relabel_archives_labels_but_keeps_pool_and_llm_cache(tmp_path, papers):
    s = store_with(tmp_path, papers)
    s.set_label(papers[0].id, True)
    s.set_label(papers[1].id, False)
    s.put_llm_score(papers[0].id, "groq/m", "v2", 8, "r")

    rnd, n = s.archive_labels()
    assert (rnd, n) == (1, 2)
    assert s.labels() == {} and len(s.unlabelled()) == 10  # start over...
    assert s.stats()["pool"] == 10 and s.get_llm_scores("groq/m", "v2")  # ...with pool + cache kept
    assert s.archived_labels(1) == {papers[0].id: 1, papers[1].id: 0}

    s.set_label(papers[0].id, False)
    assert s.archive_labels() == (2, 1)  # rounds don't overwrite each other
    assert s.archived_labels(1)[papers[0].id] == 1 and s.archived_labels(2) == {papers[0].id: 0}
    s.close()


def test_rubric_is_shown_on_every_screen(tmp_path, papers):
    s = store_with(tmp_path, papers)
    shown = []
    label_loop(s, keys("y", "n", "q"), out=shown.append, rubric="y = MAIN contribution is evaluation")
    screens = [o for o in shown if o.startswith("=")]
    assert len(screens) == 3 and all("y = MAIN contribution is evaluation" in o for o in screens)
    s.close()


def test_cli_relabel(tmp_path, papers, capsys):
    from digest.label import main

    cfg = tmp_path / "config.yaml"
    cfg.write_text("topic:\n  name: T\n  keywords: [k]\n  label_rubric: |\n    y = yes\n"
                   "database:\n  path: d.db\n  labels_path: labels.db\n", encoding="utf-8")
    with LabelStore(tmp_path / "labels.db") as s:
        s.add_to_pool(papers)
        s.set_label(papers[0].id, True)
    assert main(["--config", str(cfg), "--relabel", "--stats"]) == 0
    out = capsys.readouterr().out
    assert "Archived 1 labels as round 1" in out and "Pool 10: 0 relevant, 0 not relevant, 10 unlabelled" in out
