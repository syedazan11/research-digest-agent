"""Swappable LLM providers.

The rest of the pipeline only knows one method: complete_json(system, user) -> str.
Switching Groq <-> Gemini is a one-line change in config.yaml (rerank.provider).

Error policy:
- 429 (rate limited): wait for the server's retry-after header, then retry a few times.
- 401/403 (bad key):  raise LLMAuthError right away. Retrying can't fix a wrong key.
- anything else:      raise LLMError, and the caller decides how to fall back.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Callable, Protocol

log = logging.getLogger(__name__)

MAX_WAIT_SECONDS = 60.0  # never sleep longer than this on one retry-after
KEY_ENV = {"groq": "GROQ_API_KEY", "gemini": "GEMINI_API_KEY"}
PLACEHOLDER_KEY = "your-key-here"


class LLMError(RuntimeError):
    """The LLM call failed. Callers catch this and fall back."""


class LLMAuthError(LLMError):
    """Every further call this run would fail the same way, so callers stop early and fall back.
    Raised for a rejected API key, and (as LLMBudgetExceeded) when the run's call budget is used up."""


class LLMBudgetExceeded(LLMAuthError):
    """The per-run call budget (llm.max_calls_per_run) is used up: remaining work falls back."""


class LLMClient(Protocol):
    name: str  # provider id, e.g. "groq"
    model: str
    calls: int  # API requests made, for the run log

    def complete_json(self, system: str, user: str) -> str:
        """Send a system + user prompt and return the model's reply, which should be a JSON string."""
        ...


def _retry_after(headers, default: float) -> float:
    """Seconds to wait, from a retry-after header if the server sent one (capped at MAX_WAIT_SECONDS)."""
    value = None
    if headers is not None:
        try:
            value = headers.get("retry-after")
        except Exception:
            value = None
    try:
        wait = float(value) if value is not None else default
    except ValueError:  # retry-after can also be an HTTP date; not worth parsing, use the default
        wait = default
    return max(0.0, min(wait, MAX_WAIT_SECONDS))


class GroqLLM:
    name = "groq"

    def __init__(
        self,
        model: str = "openai/gpt-oss-120b",
        api_key: str | None = None,
        client=None,  # injectable so tests can pass a fake
        max_retries: int = 3,
        reasoning_effort: str | None = "auto",
        sleep: Callable[[float], None] = time.sleep,
    ):
        if client is None:
            from groq import Groq

            # max_retries=0: we handle 429s ourselves so we can honour retry-after and log it
            client = Groq(api_key=api_key, max_retries=0, timeout=60.0)
        self.client = client
        self.model = model
        self.max_retries = max_retries
        if reasoning_effort == "auto":
            # only reasoning models (gpt-oss) accept reasoning_effort; Llama etc. reject the request
            reasoning_effort = "low" if "gpt-oss" in model else None
        self.reasoning_effort = reasoning_effort
        self.sleep = sleep
        self.calls = 0
        self.rate_limits: dict[str, str] = {}  # latest x-ratelimit-* headers from Groq

    def _create(self, **kwargs):
        """The request, via the SDK's raw-response API when available, to keep Groq's quota headers."""
        completions = self.client.chat.completions
        raw_api = getattr(completions, "with_raw_response", None)
        if raw_api is None:  # test fakes
            return completions.create(**kwargs)
        raw = raw_api.create(**kwargs)
        try:
            self.rate_limits = {k.lower(): v for k, v in raw.headers.items() if k.lower().startswith("x-ratelimit-")}
        except Exception:  # headers are a nice-to-have, never a reason to fail a call
            pass
        return raw.parse()

    def complete_json(self, system: str, user: str) -> str:
        import groq

        for attempt in range(self.max_retries + 1):
            try:
                self.calls += 1
                extra = {"reasoning_effort": self.reasoning_effort} if self.reasoning_effort else {}
                resp = self._create(
                    model=self.model,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                    response_format={"type": "json_object"},
                    temperature=0,
                    **extra,
                )
                return resp.choices[0].message.content or ""
            except (groq.AuthenticationError, groq.PermissionDeniedError) as e:
                raise LLMAuthError(
                    f"Groq rejected the API key ({e.status_code}): check GROQ_API_KEY in .env"
                ) from e
            except groq.RateLimitError as e:
                if attempt == self.max_retries:
                    raise LLMError(f"Groq rate limit: gave up after {attempt + 1} attempts") from e
                wait = _retry_after(e.response.headers, default=5.0 * 2**attempt)
                log.warning("Groq 429 rate limit, waiting %.1fs (retry-after)", wait)
                self.sleep(wait)
            except groq.APIError as e:
                raise LLMError(f"Groq API error: {str(e)[:300]}") from e
        raise LLMError("Groq: no response")  # unreachable, keeps type checkers happy


class GeminiLLM:
    name = "gemini"

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        api_key: str | None = None,
        client=None,
        max_retries: int = 3,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if client is None:
            from google import genai

            client = genai.Client(api_key=api_key)
        self.client = client
        self.model = model
        self.max_retries = max_retries
        self.sleep = sleep
        self.calls = 0

    def complete_json(self, system: str, user: str) -> str:
        from google.genai import errors, types

        config = types.GenerateContentConfig(
            system_instruction=system, response_mime_type="application/json", temperature=0
        )
        for attempt in range(self.max_retries + 1):
            try:
                self.calls += 1
                resp = self.client.models.generate_content(model=self.model, contents=user, config=config)
                return resp.text or ""
            except errors.APIError as e:
                # Gemini reports some bad keys as 400 "API key not valid"
                if e.code in (401, 403) or (e.code == 400 and "api key" in str(e).lower()):
                    raise LLMAuthError(
                        f"Gemini rejected the API key ({e.code}): check GEMINI_API_KEY in .env"
                    ) from e
                if e.code == 429 and attempt < self.max_retries:
                    headers = getattr(getattr(e, "response", None), "headers", None)
                    wait = _retry_after(headers, default=5.0 * 2**attempt)
                    log.warning("Gemini 429 rate limit, waiting %.1fs", wait)
                    self.sleep(wait)
                    continue
                raise LLMError(f"Gemini API error {e.code}: {str(e)[:300]}") from e
        raise LLMError("Gemini: no response")


class BudgetedLLM:
    """Wraps an LLMClient with a per-run call budget shared by every topic and stage.

    Groq's free tier has per-minute and per-day limits; with several topics a bad week (many 429
    retries) could burn the day's quota. Past the budget, calls raise LLMBudgetExceeded, which
    rerank/summarize treat like an unavailable LLM: they fall back and the run is marked degraded.
    """

    def __init__(self, inner: LLMClient, max_calls: int):
        self.inner = inner
        self.max_calls = max_calls

    @property
    def name(self) -> str:
        return self.inner.name

    @property
    def model(self) -> str:
        return self.inner.model

    @property
    def calls(self) -> int:
        return self.inner.calls

    @property
    def rate_limits(self) -> dict[str, str]:
        return getattr(self.inner, "rate_limits", {})

    @property
    def exhausted(self) -> bool:
        return self.inner.calls >= self.max_calls

    def complete_json(self, system: str, user: str) -> str:
        if self.exhausted:
            raise LLMBudgetExceeded(f"LLM budget of {self.max_calls} calls per run used up (llm.max_calls_per_run)")
        return self.inner.complete_json(system, user)


def describe_rate_limits(limits: dict[str, str]) -> str:
    """'requests 980/1000 left today (resets 1m26s), tokens 6400/8000 left this minute' from Groq headers."""
    if not limits:
        return ""
    parts = []
    for kind, window in (("requests", "today"), ("tokens", "this minute")):
        left, cap = limits.get(f"x-ratelimit-remaining-{kind}"), limits.get(f"x-ratelimit-limit-{kind}")
        if left is not None and cap is not None:
            reset = limits.get(f"x-ratelimit-reset-{kind}")
            parts.append(f"{kind} {left}/{cap} left {window}" + (f" (resets {reset})" if reset else ""))
    return ", ".join(parts)


PROVIDERS = {"groq": GroqLLM, "gemini": GeminiLLM}


def make_llm(provider: str, model: str) -> LLMClient | None:
    """Build the configured provider, or return None (with a warning) if it can't be used."""
    provider = provider.lower()
    if provider not in PROVIDERS:
        raise ValueError(f"unknown LLM provider '{provider}' (choose from {sorted(PROVIDERS)})")

    env = KEY_ENV[provider]
    key = os.getenv(env, "").strip()
    if not key or key == PLACEHOLDER_KEY:
        log.warning("%s not set: LLM reranking disabled for this run (see .env.example)", env)
        return None
    try:
        return PROVIDERS[provider](model=model, api_key=key)
    except Exception as e:  # SDK missing or failed to initialise
        log.warning("Could not create %s client (%s: %s): LLM reranking disabled",
                    provider, type(e).__name__, str(e)[:200])
        return None
