from pathlib import Path

import pytest

from digest.db import Database
from digest.fetch import load_sample

FIXTURES = Path(__file__).parent / "fixtures"

# Papers in the sample file that are about the configured topic (RAG evaluation).
RELEVANT_IDS = {"sample-001", "sample-002", "sample-003", "sample-004"}

QUERY = (
    "RAG evaluation. Methods, benchmarks and metrics for evaluating retrieval-augmented "
    "generation systems, including faithfulness, hallucination detection and retrieval quality."
)


@pytest.fixture
def papers():
    return load_sample(FIXTURES / "sample_papers.json")


@pytest.fixture
def db(tmp_path):
    with Database(tmp_path / "test.db", default_topic="test-topic") as d:
        yield d


# --- fake LLM for rerank tests: scripted replies, no network ---
import json
import re

from digest.retrieval import RankedPaper

_PAPER = re.compile(r'<paper id="(P\d+)">\s*<title>(.*?)</title>', re.DOTALL)


def batch_titles(user_prompt: str) -> list[tuple[str, str]]:
    """[(label, title), ...] for the papers in a rerank prompt."""
    return _PAPER.findall(user_prompt)


def reply(scores: dict[str, float], reason: str = "fake reason") -> str:
    """JSON reply scoring labels, e.g. reply({"P1": 9, "P2": 3})."""
    return json.dumps({"scores": [{"id": k, "score": v, "reason": reason} for k, v in scores.items()]})


class FakeLLM:
    """Each entry in `replies` is a JSON string, an Exception to raise, or a function(user_prompt) -> str.
    When replies run out, `default` (if given) is used for every further call."""

    name = "fake"
    model = "fake-1"

    def __init__(self, replies=(), default=None):
        self.replies = list(replies)
        self.default = default
        self.calls = 0
        self.prompts: list[tuple[str, str]] = []

    def complete_json(self, system: str, user: str) -> str:
        self.calls += 1
        self.prompts.append((system, user))
        r = self.replies.pop(0) if self.replies else self.default
        if r is None:
            raise AssertionError("FakeLLM ran out of replies")
        if isinstance(r, Exception):
            raise r
        return r(user) if callable(r) else r


def score_by_title(table: dict[str, float], default: float = 2):
    """A FakeLLM reply function: scores each paper by a title substring, `default` otherwise."""

    def fn(user: str) -> str:
        out = {}
        for label, title in batch_titles(user):
            out[label] = next((s for key, s in table.items() if key in title), default)
        return reply(out)

    return fn


@pytest.fixture
def candidates(papers):
    """The 10 sample papers as hybrid results, hybrid rank = file order, scores 1.0, 0.9, ..."""
    return [RankedPaper(paper=p, score=round(1.0 - 0.1 * i, 2), components={"bm25": 0.0})
            for i, p in enumerate(papers)]
