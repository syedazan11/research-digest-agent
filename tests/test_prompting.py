import pytest
from pydantic import BaseModel

from conftest import FakeLLM
from digest.llm import LLMAuthError, LLMError
from digest.prompting import LLMFailed, complete_validated, escape, one_line, strip_fences


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
