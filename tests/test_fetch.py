from digest.fetch import build_query
from digest.models import strip_version


def test_build_query_combines_categories_and_keywords():
    q = build_query(["faithfulness", "RAG evaluation"], ["cs.CL", "cs.IR"])
    assert q == (
        '(cat:cs.CL OR cat:cs.IR) AND '
        '(abs:"faithfulness" OR ti:"faithfulness" OR abs:"RAG evaluation" OR ti:"RAG evaluation")'
    )


def test_build_query_handles_missing_parts():
    assert build_query([], ["cs.CL"]) == "cat:cs.CL"
    assert build_query(["rag"], []) == 'abs:"rag" OR ti:"rag"'


def test_build_query_strips_quotes():
    assert '""' not in build_query(['"rag"'], [])


def test_strip_version():
    assert strip_version("2411.01234v2") == "2411.01234"
    assert strip_version("2411.01234") == "2411.01234"
    assert strip_version("quant-ph/0201082v1") == "quant-ph/0201082"


# --- truncation warning (fake arXiv client, no network) ---
from datetime import datetime, timedelta, timezone  # noqa: E402
from types import SimpleNamespace  # noqa: E402

from digest.fetch import fetch_recent  # noqa: E402


class FakeArxiv:
    def __init__(self, ages_days):
        now = datetime.now(timezone.utc)
        self.items = [SimpleNamespace(
            get_short_id=lambda i=i: f"2611.{i:05d}v1", title=f"T{i}", summary="s", authors=[],
            published=now - timedelta(days=a), categories=["cs.AI"], pdf_url="", entry_id=f"u{i}")
            for i, a in enumerate(ages_days)]

    def results(self, search):
        return iter(self.items[: search.max_results])


def test_warns_when_the_fetch_hits_max_results(caplog):
    with caplog.at_level("WARNING"):
        got = fetch_recent(["k"], ["cs.AI"], 7, 3, client=FakeArxiv([0, 1, 2, 3, 4]))  # all within 7 days
    assert len(got) == 3 and "hit max_results=3" in caplog.text


def test_no_warning_when_the_window_ends_first(caplog):
    with caplog.at_level("WARNING"):
        got = fetch_recent(["k"], ["cs.AI"], 7, 10, client=FakeArxiv([0, 1, 9, 10]))
    assert len(got) == 2 and "max_results" not in caplog.text
