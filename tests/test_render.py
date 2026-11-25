from datetime import date

import pytest

from digest.models import Paper
from digest.render import md, render_digest
from digest.rerank import RerankedPaper
from digest.retrieval import RankedPaper
from digest.summarize import PaperSummary, Summary

S = Summary(problem="P.", method="M.", key_result="K.", why_it_matters="W.", limitations="L.")


def entry(title="A *Bold* Title", authors=6, summary=S, source="pdf", reason=None, score=9.0):
    p = Paper.from_dict({"id": "2611.1", "title": title, "abstract": "The abstract.",
                         "authors": [f"Author {i}" for i in range(authors)],
                         "published": "2026-11-20T00:00:00+00:00", "categories": [], "pdf_url": "",
                         "url": "http://arxiv.org/abs/2611.1v1"})
    item = RerankedPaper(RankedPaper(p, 0.5), hybrid_rank=2, llm_score=score, reason="On topic.", final_rank=1)
    return item, PaperSummary(p, summary, source, reason)


def render(*entries):
    return render_digest(list(entries), "LLM evaluation", 4, date(2026, 11, 25), "groq/openai/gpt-oss-120b")


def test_digest_layout():
    out = render(entry())
    assert out.startswith("# Research digest: LLM evaluation\n")
    assert "*2026-11-25 · run 4 · 1 papers · ranked by groq/openai/gpt-oss-120b*" in out
    assert r"## 1. A \*Bold\* Title" in out
    assert "**Authors:** Author 0, Author 1, Author 2, Author 3, Author 4 et al." in out
    assert "**Link:** https://arxiv.org/abs/2611.1v1" in out
    assert "**Relevance:** 9/10 - On topic." in out
    for label in ("Problem", "Method", "Key result", "Why it matters", "Limitations"):
        assert f"- **{label}:** " in out
    assert "abstract only" not in out


def test_abstract_only_and_missing_summary_are_labelled():
    out = render(entry(source="abstract", reason="no_text"), entry(summary=None))
    assert "*Summary based on the abstract only (no_text).*" in out
    assert "*Summary unavailable. Abstract:*\n\n> The abstract." in out


def test_untrusted_text_cannot_inject_markdown():
    evil = Summary(problem="![x](https://evil.example/t.png) [click](https://evil.example)",
                   method="# Heading\nnew line", key_result="K.", why_it_matters="W.", limitations="L.")
    out = render(entry(summary=evil))
    assert "![x](" not in out and "[click](" not in out
    assert r"\!\[x\](https://evil.example/t.png)" in out
    assert r"- **Method:** \# Heading new line" in out


WJ = "⁠"


@pytest.mark.parametrize("raw, safe", [
    ("thanks @octocat", f"thanks @{WJ}octocat"),
    ("cc @org/team", f"cc @{WJ}org/team"),
    ("see #123", f"see \\#{WJ}123"),
    ("fixed in octocat/Hello-World#45", f"fixed in octocat/Hello-World\\#{WJ}45"),
    ("GH-7 and gh-8", f"GH-{WJ}7 and gh-{WJ}8"),
    ("a@b.com, C# code, #hashtag, 50 @ 3", "a@b.com, C\\# code, \\#hashtag, 50 @ 3"),  # left alone
])
def test_mentions_and_issue_refs_are_defanged(raw, safe):
    # Verified against GitHub's POST /markdown renderer on 2026-11-25: a backslash does NOT stop
    # GitHub from linking @user / #1 / owner/repo#1 / GH-1; a U+2060 word joiner after @ or # does.
    assert md(raw) == safe


def test_defang_applies_to_titles_authors_and_llm_text():
    evil = Summary(problem="ping @octocat", method="see #1", key_result="K.", why_it_matters="W.",
                   limitations="L.")
    out = render(entry(title="Fixes #42 for @someone", summary=evil))
    assert "@octocat" not in out and "#1" not in out and "#42" not in out and "@someone" not in out
    assert f"@{WJ}octocat" in out and f"@{WJ}someone" in out


def test_md_escape():
    assert md("a_b *c* `d` <e> |f|") == r"a\_b \*c\* \`d\` \<e\> \|f\|"


def test_empty_digest():
    assert "No new papers this week." in render()
