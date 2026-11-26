"""Stage 4a: get the text a summary is built from: abstract + introduction + conclusion.

Why not the whole paper? Groq's free tier allows ~8K tokens a minute, and a full paper is
10-30K. The introduction states the problem and claims, the conclusion says what actually
worked; together with the abstract they carry most of what a summary needs, in ~3K tokens.

Every failure (download, not a PDF, scanned/no text, no recognisable headings) falls back to
the arXiv abstract with a recorded reason, so a bad PDF can cost quality but never the digest.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Callable

from digest.models import Paper

log = logging.getLogger(__name__)

PDF_URL = "https://export.arxiv.org/pdf/{id}"  # export.* is arXiv's mirror for programmatic access
USER_AGENT = "research-digest-agent/0.1 (+https://github.com/syedazan11/research-digest-agent)"
MAX_PDF_BYTES = 25 * 1024 * 1024
CHARS_PER_TOKEN = 4  # rough average for English; good enough for a budget, no tokenizer needed
MIN_TEXT_CHARS = 500  # less than this from a whole PDF = scanned images, not text

# fallback reasons (stored in the summaries table)
DOWNLOAD_FAILED = "download_failed"
NOT_PDF = "not_pdf"
NO_TEXT = "no_text"
NO_SECTIONS = "no_sections"

# --- heading patterns. Optional numbering: "1 ", "1. ", "IV. " (sections start at 1, never 0) ---
_SEC_NO = r"(?:[1-9]\d?|[IVX]+)"
_NUM = rf"(?:{_SEC_NO}\.?\s+)?"
INTRO = re.compile(rf"^{_NUM}introduction$", re.I)
CONCLUSION = re.compile(
    rf"^{_NUM}(?:conclusions?|concluding remarks|conclusions? and (?:future work|outlook|discussion)"
    rf"|discussion and conclusions?|summary and conclusions?)$",
    re.I,
)
DISCUSSION = re.compile(rf"^{_NUM}(?:discussion|summary)$", re.I)  # used only if no Conclusion
STOP = re.compile(rf"^{_NUM}(?:references|bibliography|appendix|appendices|acknowledge?ments?)\b", re.I)
# a top-level numbered heading like "2 Related Work" / "3. Method" (not "2.1 ...", not body text)
NEXT_SECTION = re.compile(rf"^{_SEC_NO}\.?\s+[A-Z][A-Za-z\-]*(?:\s+[A-Za-z\-:&]+){{0,7}}$")
_BARE_NUMBER = re.compile(rf"^{_SEC_NO}\.?$")
_DIGITS = re.compile(r"\d+")
REPEATED_LINE_MIN = 3  # a short line seen this often is a running header/footer ("12 Preprint")


@dataclass
class PaperText:
    text: str  # what the LLM reads
    source: str  # "pdf" or "abstract"
    fallback_reason: str | None = None  # set when source == "abstract"
    sections: list[str] = field(default_factory=list)  # e.g. ["abstract", "introduction", "conclusion"]


# --- download ---
class Throttle:
    """Keeps at least `min_interval` seconds between calls (arXiv asks for 1 request per 3s)."""

    def __init__(self, min_interval: float, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep):
        self.min_interval = min_interval
        self.clock = clock
        self.sleep = sleep
        self._last: float | None = None

    def wait(self) -> None:
        if self._last is not None:
            remaining = self.min_interval - (self.clock() - self._last)
            if remaining > 0:
                self.sleep(remaining)
        self._last = self.clock()


class PdfFetcher:
    def __init__(self, delay_seconds: float = 3.0, get: Callable[[str], bytes] | None = None,
                 throttle: Throttle | None = None, timeout: float = 30.0):
        self.get = get or self._http_get  # injectable so tests never touch the network
        self.throttle = throttle or Throttle(delay_seconds)
        self.timeout = timeout
        self.downloads = 0
        self.text_cache: dict[str, PaperText] = {}  # paper id -> extracted text, shared across topics
        self._client = None

    def _http_get(self, url: str) -> bytes:
        import httpx

        if self._client is None:
            self._client = httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=self.timeout,
                                        follow_redirects=True)
        with self._client.stream("GET", url) as resp:
            resp.raise_for_status()
            chunks, size = [], 0
            for chunk in resp.iter_bytes():
                size += len(chunk)
                if size > MAX_PDF_BYTES:
                    raise ValueError(f"PDF larger than {MAX_PDF_BYTES // 2**20} MB")
                chunks.append(chunk)
        return b"".join(chunks)

    def fetch(self, paper_id: str) -> bytes:
        self.throttle.wait()
        self.downloads += 1
        return self.get(PDF_URL.format(id=paper_id))

    def close(self) -> None:
        if self._client is not None:
            self._client.close()


# --- extraction ---
def pdf_to_text(data: bytes) -> str:
    import pymupdf

    with pymupdf.open(stream=data, filetype="pdf") as doc:
        return "\n".join(page.get_text() for page in doc)


def _clean_lines(text: str) -> list[str]:
    """Strip lines, drop empties, merge a bare section number with its title (PDFs often put
    "1" and "Introduction" on separate lines), then drop running headers/footers: short lines
    that repeat on many pages once digits are ignored ("2 Preprint", "3 Preprint", ...)."""
    raw = [ln.strip() for ln in text.splitlines()]
    raw = [ln for ln in raw if ln]
    lines: list[str] = []
    i = 0
    while i < len(raw):
        if _BARE_NUMBER.match(raw[i]) and i + 1 < len(raw) and len(raw[i + 1]) < 60:
            lines.append(f"{raw[i]} {raw[i + 1]}")
            i += 2
        else:
            lines.append(raw[i])
            i += 1

    counts: dict[str, int] = {}
    for ln in lines:
        if len(ln) < 80:
            key = _DIGITS.sub("#", ln)
            counts[key] = counts.get(key, 0) + 1
    return [ln for ln in lines if len(ln) >= 80 or counts[_DIGITS.sub("#", ln)] < REPEATED_LINE_MIN]


def _join(lines: list[str]) -> str:
    """Join body lines into prose, re-joining words hyphenated across a line break."""
    out = ""
    for ln in lines:
        if out.endswith("-") and ln[:1].islower():
            out = out[:-1] + ln
        else:
            out = f"{out} {ln}" if out else ln
    return out


def _section(lines: list[str], start: int) -> str:
    """Body of the section whose heading is lines[start], up to the next top-level heading."""
    body = []
    for ln in lines[start + 1 :]:
        if STOP.match(ln) or NEXT_SECTION.match(ln) or CONCLUSION.match(ln) or DISCUSSION.match(ln):
            break
        body.append(ln)
    return _join(body)


def _find(lines: list[str], pattern: re.Pattern, after: int = -1) -> int | None:
    return next((i for i, ln in enumerate(lines) if i > after and pattern.match(ln)), None)


def find_sections(text: str) -> tuple[str, str, str]:
    """-> (introduction, conclusion, conclusion heading kind: "conclusion"/"discussion"/"")."""
    lines = _clean_lines(text)
    i_intro = _find(lines, INTRO)
    intro = _section(lines, i_intro) if i_intro is not None else ""

    after = i_intro if i_intro is not None else -1
    i_conc = _find(lines, CONCLUSION, after)  # prefer a real Conclusion heading...
    kind = "conclusion"
    if i_conc is None:
        i_conc = _find(lines, DISCUSSION, after)  # ...use Discussion/Summary only if there is none
        kind = "discussion"
    conclusion = _section(lines, i_conc) if i_conc is not None else ""
    return intro, conclusion, kind if conclusion else ""


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars].rsplit(" ", 1)[0]
    return cut + " [...]"


def abstract_only(paper: Paper, reason: str) -> PaperText:
    return PaperText(text=f"ABSTRACT:\n{paper.abstract}", source="abstract",
                     fallback_reason=reason, sections=["abstract"])


def build_text(paper: Paper, data: bytes, max_tokens: int = 3000) -> PaperText:
    """PDF bytes -> abstract + intro + conclusion within the token budget, or an abstract fallback."""
    if not data.lstrip()[:5].startswith(b"%PDF"):
        return abstract_only(paper, NOT_PDF)
    try:
        text = pdf_to_text(data)
    except Exception:
        return abstract_only(paper, NOT_PDF)
    if len(text.strip()) < MIN_TEXT_CHARS:
        return abstract_only(paper, NO_TEXT)

    intro, conclusion, kind = find_sections(text)
    if not intro and not conclusion:
        return abstract_only(paper, NO_SECTIONS)

    # The abstract always goes in (it's short and the most reliable part); the rest of the budget
    # is split 60/40 between introduction and conclusion, or all to whichever one was found.
    budget = max_tokens * CHARS_PER_TOKEN - len(paper.abstract)
    budget = max(budget, 1000)
    if intro and conclusion:
        intro_chars = int(budget * 0.6)
        intro = _truncate(intro, intro_chars)
        conclusion = _truncate(conclusion, budget - len(intro))
    elif intro:
        intro = _truncate(intro, budget)
    else:
        conclusion = _truncate(conclusion, budget)

    parts = [f"ABSTRACT:\n{paper.abstract}"]
    sections = ["abstract"]
    if intro:
        parts.append(f"INTRODUCTION:\n{intro}")
        sections.append("introduction")
    if conclusion:
        parts.append(f"{kind.upper()}:\n{conclusion}")
        sections.append(kind)
    return PaperText(text="\n\n".join(parts), source="pdf", sections=sections)


def get_paper_text(paper: Paper, fetcher: PdfFetcher, max_tokens: int = 3000) -> PaperText:
    """Download + extract, never raising: any failure gives the abstract with a reason, logged.
    Cached on the fetcher, so a paper selected by two topics in one run is downloaded once."""
    key = f"{paper.id}:{max_tokens}"
    if key not in fetcher.text_cache:
        fetcher.text_cache[key] = _get_paper_text(paper, fetcher, max_tokens)
    return fetcher.text_cache[key]


def _get_paper_text(paper: Paper, fetcher: PdfFetcher, max_tokens: int) -> PaperText:
    if not paper.pdf_url:  # e.g. sample/offline papers: don't hit the network at all
        result = abstract_only(paper, DOWNLOAD_FAILED)
        log.warning("No PDF link for %s (%s): using the abstract", paper.id, DOWNLOAD_FAILED)
        return result
    try:
        data = fetcher.fetch(paper.id)
    except Exception as e:
        log.warning("PDF fallback for %s (%s: %s: %s): using the abstract",
                    paper.id, DOWNLOAD_FAILED, type(e).__name__, str(e)[:150])
        return abstract_only(paper, DOWNLOAD_FAILED)

    result = build_text(paper, data, max_tokens)
    if result.source == "abstract":
        log.warning("PDF fallback for %s (%s): using the abstract", paper.id, result.fallback_reason)
    else:
        log.info("PDF text for %s: %s, ~%d tokens", paper.id, "+".join(result.sections),
                 len(result.text) // CHARS_PER_TOKEN)
    return result
