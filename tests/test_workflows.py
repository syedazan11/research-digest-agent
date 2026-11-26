"""The workflow files parse and keep their safety properties (no real secrets, guards, shared lock)."""

from pathlib import Path

import yaml

WF = Path(__file__).parent.parent / ".github" / "workflows"


def load(name):
    wf = yaml.safe_load((WF / name).read_text(encoding="utf-8"))
    return wf, wf[True]  # PyYAML reads the key `on` as True


def test_tests_workflow_uses_no_secrets():
    text = (WF / "tests.yml").read_text(encoding="utf-8")
    wf, _ = load("tests.yml")
    assert "secrets." not in text and wf["permissions"] == {"contents": "read"}


def test_weekly_runs_only_where_set_up():
    wf, on = load("weekly-digest.yml")
    job = wf["jobs"]["digest"]
    assert on["schedule"][0]["cron"] == "15 4 * * 1" and "workflow_dispatch" in on
    cond = " ".join(job["if"].split())
    assert "github.event_name == 'workflow_dispatch'" in cond and "vars.DIGEST_ENABLED == 'true'" in cond
    assert job["timeout-minutes"] == 45 and wf["permissions"] == {"contents": "write", "issues": "write"}


def test_reset_workflow_is_guarded_and_shares_the_weekly_lock():
    wf, on = load("reset-data.yml")
    weekly, _ = load("weekly-digest.yml")
    job = wf["jobs"]["reset"]
    assert list(on) == ["workflow_dispatch"] and on["workflow_dispatch"]["inputs"]["confirm"]["required"]
    assert job["if"] == "github.repository != 'syedazan11/research-digest-agent'"
    assert wf["concurrency"]["group"] == weekly["concurrency"]["group"]  # can't overlap a digest run
    assert wf["permissions"] == {"contents": "write"}
    steps = [s.get("run", "") for s in job["steps"]]
    assert any('"$CONFIRM" != "RESET"' in s for s in steps)
    assert any("digest.maintenance reset --yes" in s for s in steps)
