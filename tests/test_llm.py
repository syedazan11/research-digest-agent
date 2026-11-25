"""LLM providers with fake SDK clients: no network, no API keys, no real sleeping."""

import logging
from types import SimpleNamespace

import groq
import httpx
import pytest
from google.genai import errors as genai_errors

from digest.llm import GeminiLLM, GroqLLM, LLMAuthError, LLMError, _retry_after, make_llm

REQ = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")


def groq_error(cls, status, headers=None):
    return cls(f"error {status}", response=httpx.Response(status, headers=headers or {}, request=REQ), body=None)


class FakeGroq:
    """Mimics client.chat.completions.create(...). `outcomes`: Exceptions to raise or strings to return."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.kwargs = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.kwargs.append(kwargs)
        o = self.outcomes.pop(0)
        if isinstance(o, Exception):
            raise o
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=o))])


def make_groq(outcomes):
    sleeps = []
    fake = FakeGroq(outcomes)
    return GroqLLM(model="m", client=fake, sleep=sleeps.append), fake, sleeps


# --- Groq ---
def test_groq_request_uses_json_mode_and_settings():
    llm, fake, _ = make_groq(['{"scores": []}'])
    assert llm.complete_json("sys", "usr") == '{"scores": []}'
    kw = fake.kwargs[0]
    assert kw["response_format"] == {"type": "json_object"}
    assert kw["temperature"] == 0 and kw["reasoning_effort"] == "low" and kw["model"] == "m"
    assert kw["messages"] == [{"role": "system", "content": "sys"}, {"role": "user", "content": "usr"}]


def test_groq_429_waits_for_retry_after(caplog):
    llm, _, sleeps = make_groq([groq_error(groq.RateLimitError, 429, {"retry-after": "7"}), "{}"])
    with caplog.at_level(logging.WARNING):
        assert llm.complete_json("s", "u") == "{}"
    assert sleeps == [7.0]
    assert llm.calls == 2
    assert "429" in caplog.text


def test_groq_429_gives_up_with_llm_error():
    llm, _, sleeps = make_groq([groq_error(groq.RateLimitError, 429)] * 4)
    with pytest.raises(LLMError, match="rate limit"):
        llm.complete_json("s", "u")
    assert len(sleeps) == 3  # max_retries=3: 4 attempts, 3 waits
    assert sleeps == [5.0, 10.0, 20.0]  # no header -> exponential backoff


def test_groq_auth_error_is_clear_and_not_retried():
    llm, _, sleeps = make_groq([groq_error(groq.AuthenticationError, 401)])
    with pytest.raises(LLMAuthError, match="GROQ_API_KEY"):
        llm.complete_json("s", "u")
    assert llm.calls == 1 and sleeps == []


def test_groq_other_api_error_becomes_llm_error():
    llm, _, _ = make_groq([groq_error(groq.InternalServerError, 500)])
    with pytest.raises(LLMError):
        llm.complete_json("s", "u")


def test_retry_after_parsing():
    assert _retry_after({"retry-after": "2.5"}, default=9) == 2.5
    assert _retry_after({"retry-after": "9999"}, default=9) == 60.0  # capped
    assert _retry_after({"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"}, default=9) == 9
    assert _retry_after(None, default=4) == 4


# --- Gemini ---
class FakeGemini:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.models = SimpleNamespace(generate_content=self._gen)

    def _gen(self, model, contents, config):
        assert config.response_mime_type == "application/json"
        o = self.outcomes.pop(0)
        if isinstance(o, Exception):
            raise o
        return SimpleNamespace(text=o)


def gemini_error(code, msg="err"):
    return genai_errors.APIError(code, {"error": {"code": code, "message": msg, "status": "X"}})


def test_gemini_success_and_429_retry():
    sleeps = []
    llm = GeminiLLM(model="g", client=FakeGemini([gemini_error(429), '{"scores": []}']), sleep=sleeps.append)
    assert llm.complete_json("s", "u") == '{"scores": []}'
    assert sleeps == [5.0]


@pytest.mark.parametrize("err", [gemini_error(401), gemini_error(400, "API key not valid. Please pass a valid API key.")])
def test_gemini_auth_error(err):
    llm = GeminiLLM(model="g", client=FakeGemini([err]), sleep=lambda s: None)
    with pytest.raises(LLMAuthError, match="GEMINI_API_KEY"):
        llm.complete_json("s", "u")


# --- factory ---
@pytest.mark.parametrize("value", [None, "", "your-key-here"])
def test_make_llm_without_key_returns_none(monkeypatch, caplog, value):
    if value is None:
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
    else:
        monkeypatch.setenv("GROQ_API_KEY", value)
    with caplog.at_level(logging.WARNING):
        assert make_llm("groq", "openai/gpt-oss-120b") is None
    assert "GROQ_API_KEY not set" in caplog.text


def test_make_llm_builds_provider(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-real")  # constructing the client makes no request
    llm = make_llm("Groq", "openai/gpt-oss-120b")
    assert isinstance(llm, GroqLLM) and llm.model == "openai/gpt-oss-120b"


def test_make_llm_unknown_provider():
    with pytest.raises(ValueError):
        make_llm("openai", "gpt")
