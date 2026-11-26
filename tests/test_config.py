from pathlib import Path

import pytest

from digest.config import load_config

ROOT = Path(__file__).parent.parent
BASE = """
topic:
  name: RAG evaluation
  description: Evaluating retrieval-augmented generation.
  keywords: [RAG evaluation]
  categories: [cs.CL]
database:
  path: test.db
"""


def write(tmp_path, extra=""):
    p = tmp_path / "config.yaml"
    p.write_text(BASE + extra, encoding="utf-8")
    return p


def test_project_config_has_rerank_section():
    rc = load_config(ROOT / "config.yaml").rerank
    assert rc.enabled and rc.provider == "groq"
    assert rc.model == "openai/gpt-oss-120b" and rc.top_n == 5 and rc.batch_size == 5


def test_rerank_defaults_when_section_missing(tmp_path):
    rc = load_config(write(tmp_path)).rerank
    assert rc.enabled and rc.provider == "groq" and rc.top_n == 5


def test_rerank_custom_values(tmp_path):
    cfg = load_config(write(tmp_path, "rerank:\n  enabled: false\n  provider: Gemini\n  model: x\n  top_n: 3\n  batch_size: 4\n"))
    assert (cfg.rerank.enabled, cfg.rerank.provider, cfg.rerank.model, cfg.rerank.top_n, cfg.rerank.batch_size) == (
        False, "gemini", "x", 3, 4)


@pytest.mark.parametrize("extra", ["rerank:\n  provider: openai\n", "rerank:\n  top_n: 0\n"])
def test_rerank_rejects_bad_values(tmp_path, extra):
    with pytest.raises(ValueError):
        load_config(write(tmp_path, extra))


def test_summarize_section(tmp_path):
    assert load_config(ROOT / "config.yaml").summarize.max_input_tokens == 3000
    sc = load_config(write(tmp_path, "summarize:\n  enabled: false\n  max_input_tokens: 2000\n")).summarize
    assert (sc.enabled, sc.max_input_tokens, sc.pdf_delay_seconds) == (False, 2000, 3.0)


@pytest.mark.parametrize("extra", ["summarize:\n  pdf_delay_seconds: 1\n", "summarize:\n  max_input_tokens: 100\n"])
def test_summarize_rejects_bad_values(tmp_path, extra):
    with pytest.raises(ValueError):
        load_config(write(tmp_path, extra))


def test_deliver_section(tmp_path):
    d = load_config(ROOT / "config.yaml").deliver
    assert (d.provider, d.close_previous, d.comment_delay_seconds, d.feedback_days, d.feedback_users) == (
        "github", True, 1.0, 30, [])
    d = load_config(write(tmp_path, "deliver:\n  provider: None\n  feedback_users: [me]\n")).deliver
    assert d.provider == "none" and d.feedback_users == ["me"]


@pytest.mark.parametrize("extra", ["deliver:\n  provider: telegram\n", "deliver:\n  comment_delay_seconds: 0\n"])
def test_deliver_rejects_bad_values(tmp_path, extra):
    with pytest.raises(ValueError):
        load_config(write(tmp_path, extra))


def test_day6_config(tmp_path):
    cfg = load_config(ROOT / "config.yaml")
    assert cfg.rerank.prompt_version in {"v1", "v2", "v3"}
    assert cfg.labels_path.name == "labels.db"
    c = load_config(write(tmp_path, "ranking:\n  weights:\n    feedback: 0.3\nrerank:\n  prompt_version: v3\n"))
    assert c.ranking.feedback_weight == 0.3 and c.rerank.prompt_version == "v3"
    with pytest.raises(ValueError):
        load_config(write(tmp_path, "rerank:\n  prompt_version: v9\n"))


def test_production_feedback_share_matches_the_eval():
    """The eval mixes BM25 and feedback as (1 - w, w) with w = 0.2. Production normalizes the weights of
    the active retrievers, so with embeddings off the feedback share must be 0.2 as well."""
    cfg = load_config(ROOT / "config.yaml")
    assert cfg.embeddings.provider == "none"
    share = cfg.ranking.feedback_weight / (cfg.ranking.bm25_weight + cfg.ranking.feedback_weight)
    assert share == pytest.approx(0.2)
    assert cfg.rerank.prompt_version == "v1"


# --- Day 7: multiple topics ---
TOPICS = """
topics:
  - name: LLM evaluation
    keywords: [LLM evaluation]
  - name: AI agents
    slug: ai-agents
    keywords: [LLM agent]
    top_n: 3
database:
  path: test.db
"""


def test_project_config_has_two_topics_with_llm_evaluation_first():
    cfg = load_config(ROOT / "config.yaml")
    assert [t.slug for t in cfg.topics] == ["llm-evaluation", "ai-agents"]
    assert cfg.topic.slug == "llm-evaluation"  # existing data was migrated to topics[0]
    assert "cs.MA" in cfg.get_topic("ai-agents").categories and cfg.get_topic("ai-agents").label_rubric


def test_topics_list(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(TOPICS, encoding="utf-8")
    cfg = load_config(p)
    assert [t.slug for t in cfg.topics] == ["llm-evaluation", "ai-agents"]  # first slug derived from the name
    assert cfg.top_n(cfg.topics[0]) == 5 and cfg.top_n(cfg.topics[1]) == 3
    with pytest.raises(ValueError, match="unknown topic"):
        cfg.get_topic("nope")


def test_single_topic_config_still_works(tmp_path):
    cfg = load_config(write(tmp_path))  # BASE uses the old `topic:` key
    assert len(cfg.topics) == 1 and cfg.topic.slug == "rag-evaluation" and cfg.topic.name == "RAG evaluation"


@pytest.mark.parametrize("bad, match", [
    ("topics:\n  - name: A\n    keywords: [k]\n  - name: A\n    keywords: [k]\n", "unique"),
    ("topics:\n  - name: A\n    slug: Not_A_Slug\n    keywords: [k]\n", "lowercase"),
    ("topics:\n  - name: A\n    keywords: [k]\ntopic:\n  name: B\n  keywords: [k]\n", "either"),
    ("topics:\n  - name: A\n", "keyword"),
    ("topics:\n  - name: A\n    keywords: [k]\n    top_n: 0\n", "top_n"),
    ("topics:\n  - name: A\n    keywords: [k]\nllm:\n  max_calls_per_run: 0\n", "max_calls"),
])
def test_topic_validation(tmp_path, bad, match):
    p = tmp_path / "c.yaml"
    p.write_text(bad, encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        load_config(p)


def test_example_config_loads_and_differs_from_mine():
    ex = load_config(ROOT / "config.example.yaml")
    assert [t.slug for t in ex.topics] == ["rag-evaluation", "robot-learning"]
    assert ex.top_n(ex.topics[1]) == 3 and ex.llm.max_calls_per_run == 60
    assert all(t.label_rubric for t in ex.topics)
    mine = {t.slug for t in load_config(ROOT / "config.yaml").topics}
    assert not mine & {t.slug for t in ex.topics}  # nobody copies my topics by accident
