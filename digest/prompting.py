"""Shared helpers for LLM stages (rerank, summarize): one injection guard, one validate-and-retry loop.

Keeping these in one place means the two stages can't drift apart on safety rules.
"""

from __future__ import annotations

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
            return model.model_validate_json(strip_fences(raw))
        except ValidationError as e:
            last_error = f"invalid JSON: {e.errors(include_url=False)[:3]}"
            log.warning("LLM reply failed validation%s", ", retrying with the error" if attempt == 0 else "")
            prompt = user + RETRY_SUFFIX.format(error=str(e)[:500], example=example)
    raise LLMFailed(last_error)
