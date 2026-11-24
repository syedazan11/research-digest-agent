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
