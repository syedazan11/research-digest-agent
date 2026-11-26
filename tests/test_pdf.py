"""PDF download + section extraction. PDFs are generated in the test with PyMuPDF; no network."""

import logging

import pymupdf
import pytest

from digest.models import Paper
from digest.pdf import (
    CHARS_PER_TOKEN, DOWNLOAD_FAILED, NO_SECTIONS, NO_TEXT, NOT_PDF, PDF_URL, USER_AGENT,
    PdfFetcher, Throttle, build_text, find_sections, get_paper_text,
)

FILLER = "This sentence is ordinary body text about evaluating language models carefully. "


def make_pdf(blocks: list[str]) -> bytes:
    """One page per block; text wraps inside the page."""
    doc = pymupdf.open()
    for block in blocks:
        page = doc.new_page()
        assert page.insert_textbox(pymupdf.Rect(50, 50, 550, 800), block, fontsize=8) >= 0, "text overflow"
    data = doc.tobytes()
    doc.close()
    return data


def paper(pdf_url="https://arxiv.org/pdf/2611.00001v1", abstract="We study judges."):
    return Paper.from_dict({"id": "2611.00001", "title": "T", "abstract": abstract, "authors": ["A"],
                            "published": "2026-11-20T00:00:00+00:00", "categories": ["cs.CL"],
                            "pdf_url": pdf_url, "url": ""})


STANDARD = [
    "A Paper Title\nSome Author\nAbstract\nWe study judges.\n1\nIntroduction\n"
    + "Intro starts here. " + FILLER * 8 + "Intro ends here.\n"
    "2 Related Work\nRELATED-WORK-TEXT " + FILLER * 3,
    "3 Method\nMETHOD-TEXT " + FILLER * 3
    + "\n5 Discussion\nDISCUSSION-TEXT " + FILLER * 2
    + "\n6 Conclusion\nConclusion starts here. " + FILLER * 3 + "Conclusion ends here.\n"
    "References\n[1] Someone. A cited paper. 2024.",
]


# --- section heuristic ---
def test_finds_intro_and_prefers_conclusion_over_discussion():
    pt = build_text(paper(), make_pdf(STANDARD))
    assert pt.source == "pdf" and pt.fallback_reason is None
    assert pt.sections == ["abstract", "introduction", "conclusion"]
    assert "Intro starts here." in pt.text and "Intro ends here." in pt.text
    assert "Conclusion starts here." in pt.text and "Conclusion ends here." in pt.text
    for absent in ("RELATED-WORK-TEXT", "METHOD-TEXT", "DISCUSSION-TEXT", "cited paper"):
        assert absent not in pt.text
    assert pt.text.startswith("ABSTRACT:\nWe study judges.")


def test_discussion_used_only_when_no_conclusion():
    text = "1 Introduction\nIntro text. " + FILLER + "\n4 Discussion\nDiscussion body. " + FILLER + "\nReferences\n[1] x"
    intro, conclusion, kind = find_sections(text)
    assert kind == "discussion" and conclusion.startswith("Discussion body.")
    assert "Discussion body" not in intro


@pytest.mark.parametrize("heading", ["6 Conclusions", "VI. CONCLUSION", "Concluding Remarks",
                                     "7. Conclusion and Future Work", "Discussion and Conclusions"])
def test_conclusion_heading_variants(heading):
    text = f"1 Introduction\nIntro.\n5 Discussion\nDisc.\n{heading}\nThe end.\nAcknowledgments\nThanks."
    _, conclusion, kind = find_sections(text)
    assert (kind, conclusion) == ("conclusion", "The end.")


def test_running_headers_and_figure_numbers_dont_end_a_section():
    # Real failure from arXiv 2611.28940: "0 Favors A" (figure text) and "2 Preprint"
    # (page number + running header) both looked like section headings and cut the intro short.
    text = "\n".join([
        "1", "Preprint", "1 Introduction", "Intro part one.", "Favors B", "0 Favors A", "Intro part two.",
        "2", "Preprint", "Intro part three.",
        "3", "Preprint", "2 Background", "Background text.",
        "4", "Preprint", "7 Conclusion", "Done.",
    ])
    intro, conclusion, _ = find_sections(text)
    assert intro == "Intro part one. Favors B 0 Favors A Intro part two. Intro part three."
    assert conclusion == "Done."


def test_hyphenated_line_breaks_are_rejoined():
    intro, _, _ = find_sections("1 Introduction\nWe evalu-\nate judges.\n2 Method\nx")
    assert intro == "We evaluate judges."


def test_token_budget_is_respected():
    long = ["1 Introduction\n" + FILLER * 60, "7 Conclusion\n" + FILLER * 60 + "\nReferences\n[1] x"]
    pt = build_text(paper(), make_pdf(long), max_tokens=500)
    assert pt.source == "pdf"
    assert len(pt.text) <= 500 * CHARS_PER_TOKEN + 100  # + section labels
    assert "[...]" in pt.text and "CONCLUSION:" in pt.text  # both sections kept, both truncated


# --- fallbacks ---
def test_not_a_pdf():
    pt = build_text(paper(), b"<html>Rate limit exceeded</html>")
    assert (pt.source, pt.fallback_reason) == ("abstract", NOT_PDF)
    assert pt.text == "ABSTRACT:\nWe study judges."


def test_corrupt_pdf():
    assert build_text(paper(), b"%PDF-1.7 garbage garbage").fallback_reason == NOT_PDF


def test_scanned_pdf_has_no_text():
    assert build_text(paper(), make_pdf(["", ""])).fallback_reason == NO_TEXT


def test_no_recognisable_headings():
    pt = build_text(paper(), make_pdf([FILLER * 20]))
    assert (pt.source, pt.fallback_reason) == ("abstract", NO_SECTIONS)


def test_download_failure_falls_back_and_logs(caplog):
    def boom(url):
        raise ConnectionError("arXiv unreachable")

    fetcher = PdfFetcher(get=boom, throttle=Throttle(0))
    with caplog.at_level(logging.WARNING):
        pt = get_paper_text(paper(), fetcher)
    assert (pt.source, pt.fallback_reason) == ("abstract", DOWNLOAD_FAILED)
    assert "download_failed" in caplog.text and "2611.00001" in caplog.text


def test_no_pdf_link_never_touches_the_network():
    def must_not_call(url):
        raise AssertionError("no download expected")

    pt = get_paper_text(paper(pdf_url=""), PdfFetcher(get=must_not_call, throttle=Throttle(0)))
    assert pt.fallback_reason == DOWNLOAD_FAILED


# --- fetcher ---
def test_fetcher_uses_export_arxiv_and_throttles():
    urls, sleeps, now = [], [], [100.0]
    throttle = Throttle(3.0, clock=lambda: now[0], sleep=lambda s: (sleeps.append(s), now.__setitem__(0, now[0] + s)))
    fetcher = PdfFetcher(get=lambda url: urls.append(url) or b"%PDF", throttle=throttle)

    fetcher.fetch("2611.00001")
    now[0] += 1.0  # 1s later: must wait 2s more
    fetcher.fetch("2611.00002")
    now[0] += 5.0  # 5s later: no wait
    fetcher.fetch("2611.00003")

    assert urls[0] == "https://export.arxiv.org/pdf/2611.00001" == PDF_URL.format(id="2611.00001")
    assert sleeps == [pytest.approx(2.0)]
    assert fetcher.downloads == 3


def test_user_agent_identifies_the_project():
    assert USER_AGENT.startswith("research-digest-agent/") and "(+https://" in USER_AGENT


def test_a_paper_picked_by_two_topics_is_downloaded_once():
    http_calls = []
    fetcher = PdfFetcher(get=lambda url: http_calls.append(url) or make_pdf(STANDARD), throttle=Throttle(0))
    first = get_paper_text(paper(), fetcher)
    second = get_paper_text(paper(), fetcher)  # same run, another topic
    assert first is second and len(http_calls) == 1 and fetcher.downloads == 1



def test_user_agent_never_claims_to_be_another_copy(monkeypatch):
    from digest.identity import SOFTWARE_URL, user_agent

    monkeypatch.delenv("DIGEST_PROJECT_URL", raising=False)
    monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
    assert user_agent().endswith(f"(+{SOFTWARE_URL})")  # local, no config: the software's homepage
    monkeypatch.setenv("GITHUB_REPOSITORY", "someone/their-digest")  # a template copy in Actions
    assert user_agent().endswith("(+https://github.com/someone/their-digest)")
    monkeypatch.setenv("DIGEST_PROJECT_URL", "https://example.org/me")  # config.yaml project.url wins
    assert user_agent().endswith("(+https://example.org/me)")
