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
