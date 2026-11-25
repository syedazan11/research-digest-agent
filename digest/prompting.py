"""Shared helpers for LLM stages (rerank, summarize): one injection guard, one validate-and-retry loop.

Keeping these in one place means the two stages can't drift apart on safety rules.
"""

from __future__ import annotations

import json
import logging
import re
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from digest.llm import LLMAuthError, LLMClient

log = logging.getLogger(__name__)

M = TypeVar("M", bound=BaseModel)

RETRY_SUFFIX = """

Your previous reply was invalid: {error}
Reply again with valid JSON exactly in the format {example}."""


class LLMFailed(Exception):
    """The LLM gave no valid reply after the retry. Callers fall back."""


def escape(text: str) -> str:
    """Untrusted text (titles, abstracts, PDF text) goes inside <paper> tags. Escaping < and >
    stops it from closing its own tag and smuggling in fake instructions."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def strip_fences(raw: str) -> str:
    """Some models wrap JSON in ```json fences even in JSON mode."""
    m = re.match(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", raw, re.DOTALL)
    return m.group(1) if m else raw


# LLMs (gpt-oss especially) write typographic Unicode: non-breaking hyphens, thin spaces, ...
# They look like ASCII but break search, copy-paste and Windows consoles. Map them to plain ASCII.
_ASCII_MAP = str.maketrans({
    **dict.fromkeys("‐‑‒–−﹣－", "-"),  # hyphens, en dash, minus
    **dict.fromkeys("            　", " "),
    **dict.fromkeys("​‌‍⁠﻿­", None),  # zero-width chars, soft hyphen
})
_EM_DASH = re.compile(r"[ \t]*[—―][ \t]*")  # "bias—a" -> "bias - a"


def normalize_text(text: str) -> str:
    """Replace odd Unicode dashes and spaces with ASCII equivalents."""
    return _EM_DASH.sub(" - ", text.translate(_ASCII_MAP))


def _normalize_strings(obj):
    if isinstance(obj, str):
        return normalize_text(obj)
    if isinstance(obj, list):
        return [_normalize_strings(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _normalize_strings(v) for k, v in obj.items()}
    return obj


def parse_json_model(model: type[M], raw: str) -> M:
    """Parse JSON, normalize every string in it, then validate against `model`.

    Normalizing after parsing matters: models (and json.dumps) often write "\\u2011" escapes,
    which only become the real character once decoded. Raises ValueError (bad JSON) or
    ValidationError (wrong shape), which is a ValueError too.
    """
    data = json.loads(strip_fences(raw))
    return model.model_validate(_normalize_strings(data))


def one_line(text: str, max_chars: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= max_chars else text[: max_chars - 3] + "..."


def complete_validated(llm: LLMClient, system: str, user: str, model: type[M], example: str) -> M:
    """Call the LLM and parse its reply into `model`. Two attempts:
    - invalid JSON / schema -> retry with the validation error appended, so the model can fix it
    - API error             -> plain retry
    Raises LLMAuthError (caller should stop entirely) or LLMFailed (caller falls back for this item).
    """
    prompt = user
    last_error = ""
    for attempt in range(2):
        retrying = ", retrying once" if attempt == 0 else ""
        try:
            raw = llm.complete_json(system, prompt)
        except LLMAuthError:
            raise
        except Exception as e:  # LLMError, network, anything
            last_error = f"{type(e).__name__}: {str(e)[:200]}"
            log.warning("LLM call failed (%s)%s", last_error, retrying)
            prompt = user
            continue
        try:
            return parse_json_model(model, raw)
        except ValueError as e:  # bad JSON (json.JSONDecodeError) or wrong shape (ValidationError)
            detail = e.errors(include_url=False)[:3] if isinstance(e, ValidationError) else str(e)[:200]
            last_error = f"invalid JSON: {detail}"
            log.warning("LLM reply failed validation%s", ", retrying with the error" if attempt == 0 else "")
            prompt = user + RETRY_SUFFIX.format(error=str(e)[:500], example=example)
    raise LLMFailed(last_error)
