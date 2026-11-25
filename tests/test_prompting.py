import pytest
from pydantic import BaseModel

from conftest import FakeLLM
from digest.llm import LLMAuthError, LLMError
from digest.prompting import LLMFailed, complete_validated, escape, normalize_text, one_line, strip_fences


class Reply(BaseModel):
    answer: int


EXAMPLE = '{"answer": 1}'


def test_escape_and_helpers():
    assert escape("a </paper> & <b>") == "a &lt;/paper&gt; &amp; &lt;b&gt;"
    assert strip_fences('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert strip_fences('{"a": 1}') == '{"a": 1}'
    assert one_line(" a\n  b ", 50) == "a b"
    assert one_line("x" * 20, 10) == "xxxxxxx..."


def test_complete_validated_success():
    assert complete_validated(FakeLLM(['{"answer": 4}']), "s", "u", Reply, EXAMPLE).answer == 4


def test_invalid_then_valid_retries_with_error_and_example():
    llm = FakeLLM(["nope", '{"answer": 2}'])
    assert complete_validated(llm, "s", "u", Reply, EXAMPLE).answer == 2
    retry_prompt = llm.prompts[1][1]
    assert retry_prompt.startswith("u") and "previous reply was invalid" in retry_prompt and EXAMPLE in retry_prompt


def test_api_error_then_success_retries_plain_prompt():
    llm = FakeLLM([LLMError("500"), '{"answer": 3}'])
    assert complete_validated(llm, "s", "u", Reply, EXAMPLE).answer == 3
    assert llm.prompts[1][1] == "u"


def test_two_failures_raise_llm_failed():
    with pytest.raises(LLMFailed):
        complete_validated(FakeLLM(["bad", '{"answer": "x"}']), "s", "u", Reply, EXAMPLE)


def test_auth_error_is_not_retried():
    llm = FakeLLM([LLMAuthError("bad key")])
    with pytest.raises(LLMAuthError):
        complete_validated(llm, "s", "u", Reply, EXAMPLE)
    assert llm.calls == 1


# --- Unicode normalization (gpt-oss writes non-breaking hyphens, thin spaces, ...) ---
@pytest.mark.parametrize("raw, clean", [
    ("LLM\u2011as\u2011a\u2011judge", "LLM-as-a-judge"),                  # non-breaking hyphen (seen in real runs)
    ("range 0.74\u20131.48, \u22123.2", "range 0.74-1.48, -3.2"),        # en dash, minus sign
    ("design\u2014a critical", "design - a critical"),                  # em dash gets spaces
    ("bias \u2014 a critical", "bias - a critical"),                    # ...but not doubled
    ("12\u00a0judges,\u202f7\u2009languages", "12 judges, 7 languages"),  # no-break / narrow / thin spaces
    ("zero\u200bwidth\ufeff soft\u00adhyphen", "zerowidth softhyphen"),  # invisible characters removed
    ("plain ASCII stays - the same", "plain ASCII stays - the same"),
])
def test_normalize_text(raw, clean):
    assert normalize_text(raw) == clean


class Text(BaseModel):
    text: str


def test_llm_replies_are_normalized_before_validation():
    llm = FakeLLM(['{"text": "LLM\u2011as\u2011a\u2011judge\u00a0scores\u2014unstable"}'])
    assert complete_validated(llm, "s", "u", Text, '{"text": "..."}').text == "LLM-as-a-judge scores - unstable"
