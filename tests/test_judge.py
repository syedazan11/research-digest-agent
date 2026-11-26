"""Summary judge: sentence counting, length rules, verdict parsing, report. Fake LLM, no network."""

import json

from conftest import FakeLLM
from digest.judge import count_sentences, judge_one, length_issues, main, render
from digest.llm import LLMAuthError
from digest.summarize import FIELDS, Summary

GOOD = Summary(problem="Judges are unstable.", method="They vary score ranges, e.g. 1-5 vs. 0-100.",
               key_result="Scores drop from 97.4 to 89.9.", why_it_matters="It matters.",
               limitations="Not stated in the provided text.")


def verdicts(**overrides):
    return json.dumps({"fields": [{"field": f, "verdict": overrides.get(f, "supported"), "note": "x"}
                                  for f in FIELDS]})


def test_count_sentences_ignores_abbreviations_and_decimals():
    assert count_sentences("One sentence with e.g. an example and 0.74 points.") == 1
    assert count_sentences("Smith et al. show it. Then it works.") == 2
    assert count_sentences("A. B. C.") == 3
    assert count_sentences("") == 0


def test_length_issues():
    assert length_issues(GOOD) == []
    long = GOOD.model_copy(update={"method": "One. Two. Three."})
    assert length_issues(long) == ["method: 3 sentences"]


def test_judge_one_parses_verdicts(papers):
    llm = FakeLLM([verdicts(key_result="partly")])
    j = judge_one(llm, papers[0], "source text", "pdf", GOOD)
    assert j.supported == 4 and j.verdicts["key_result"].verdict == "partly" and not j.error
    system, user = llm.prompts[0]
    assert "untrusted" in system and "<text>\nsource text\n</text>" in user and "97.4" in user


def test_judge_one_failures(papers):
    j = judge_one(FakeLLM(["nope", "still nope"]), papers[0], "t", "pdf", GOOD)
    assert j.error and j.verdicts == {}
    partial = json.dumps({"fields": [{"field": "problem", "verdict": "supported"}]})
    assert "judge covered" in judge_one(FakeLLM([partial]), papers[0], "t", "pdf", GOOD).error
    try:
        judge_one(FakeLLM([LLMAuthError("bad")]), papers[0], "t", "pdf", GOOD)
        raise AssertionError("auth errors must propagate")
    except LLMAuthError:
        pass


def test_render(papers):
    rs = [judge_one(FakeLLM([verdicts()]), papers[0], "t", "pdf", GOOD),
          judge_one(FakeLLM([verdicts(method="unsupported")]), papers[1], "t", "abstract",
                    GOOD.model_copy(update={"method": "A. B. C."}))]
    out = render(rs, "groq/summ", "groq/judge", "2026-11-26")
    assert "| supported | 9 | 90% |" in out and "| unsupported | 1 | 10% |" in out
    assert "1/2" in out and "method (unsupported: x)" in out and "method: 3 sentences" in out


def test_main_judges_cached_summaries(tmp_path, papers):
    from digest.db import Database
    from digest.pdf import PdfFetcher, Throttle

    cfg = tmp_path / "config.yaml"
    cfg.write_text("topic:\n  name: T\n  keywords: [k]\ndatabase:\n  path: data/digest.db\n", encoding="utf-8")
    with Database(tmp_path / "data" / "digest.db") as db:
        db.save_papers(papers[:2])
        for p in papers[:2]:  # sample papers have no PDF link, so both are abstract-based
            db.put_summary(p.id, "groq/openai/gpt-oss-120b", GOOD.model_dump_json(), "abstract", "download_failed")
    before = (tmp_path / "data" / "digest.db").read_bytes()

    llm = FakeLLM(default=lambda user: verdicts())
    out = tmp_path / "s.md"
    fetcher = PdfFetcher(get=lambda url: (_ for _ in ()).throw(AssertionError("no download")), throttle=Throttle(0))
    assert main(["--config", str(cfg), "--out", str(out)], make_llm=lambda p, m: llm, fetcher=fetcher) == 0
    assert llm.calls == 2 and "| supported | 10 | 100% |" in out.read_text(encoding="utf-8")
    assert "ABSTRACT:" in llm.prompts[0][1]
    assert (tmp_path / "data" / "digest.db").read_bytes() == before  # read only
