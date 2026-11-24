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
    with Database(tmp_path / "test.db") as d:
        yield d
