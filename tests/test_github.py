"""GitHubClient against the in-memory fake: errors, rate limits, pagination. No network."""

import logging

import httpx
import pytest

from digest.github import (
    MAX_WAIT_SECONDS, SECONDARY_LIMIT_WAIT, GitHubAuthError, GitHubClient, GitHubError, GitHubNotFound,
    USER_AGENT, make_github,
)
from fake_github import REPO, TOKEN, FakeGitHub


def once(response):
    """A `before` hook that injects `response` on the first request only."""
    state = {"done": False}

    def hook(request):
        if not state["done"]:
            state["done"] = True
            return response
        return None

    return hook


def test_headers_and_basic_calls():
    gh = FakeGitHub()
    seen = {}
    gh.before = lambda req: seen.update(req.headers) or None
    c = gh.client()
    c.check_access()
    issue = c.create_issue("T", "B", ["digest"])
    comment = c.create_comment(issue["number"], "hello")
    assert c.list_comments(issue["number"])[0]["id"] == comment["id"]
    assert seen["authorization"] == f"Bearer {TOKEN}"
    assert seen["user-agent"].startswith("research-digest-agent/") and seen["x-github-api-version"] == "2022-11-28"


def test_label_already_existing_is_fine():
    gh = FakeGitHub()
    c = gh.client()
    c.ensure_label("digest")
    c.ensure_label("digest")  # 422 from GitHub, swallowed
    assert gh.labels == {"digest"}


def test_pagination_reads_every_page():
    gh = FakeGitHub()
    c = gh.client()
    n = c.create_issue("T", "B", [])["number"]
    for i in range(230):
        c.create_comment(n, f"c{i}")
    assert len(c.list_comments(n)) == 230
    assert gh.count("GET", rf"/repos/{REPO}/issues/{n}/comments") == 3


# --- errors ---
def test_bad_token_is_a_clear_auth_error_and_token_never_leaks():
    gh = FakeGitHub()
    gh.before = lambda req: httpx.Response(401, json={"message": "Bad credentials"})
    with pytest.raises(GitHubAuthError, match="invalid or expired") as err:
        gh.client().check_access()
    assert TOKEN not in str(err.value)
    assert len(gh.log) == 1  # not retried


def test_missing_permission_is_an_auth_error():
    gh = FakeGitHub()
    gh.before = lambda req: httpx.Response(403, json={"message": "Resource not accessible by personal access token"})
    with pytest.raises(GitHubAuthError, match="Issues: Read and write"):
        gh.client().create_issue("T", "B", [])


def test_unknown_repo_is_an_auth_error():
    gh = FakeGitHub()
    gh.before = lambda req: httpx.Response(404, json={"message": "Not Found"})
    with pytest.raises(GitHubAuthError, match="not found"):
        gh.client().check_access()


def test_missing_comment_is_not_found_not_auth():
    with pytest.raises(GitHubNotFound):
        FakeGitHub().client().list_reactions(999999)


# --- rate limits ---
def test_secondary_rate_limit_waits_for_retry_after(caplog):
    gh = FakeGitHub()
    gh.before = once(httpx.Response(403, headers={"retry-after": "7"},
                                    json={"message": "You have exceeded a secondary rate limit"}))
    sleeps = []
    c = gh.client(sleep=sleeps.append)
    with caplog.at_level(logging.WARNING):
        c.create_issue("T", "B", [])
    assert sleeps == [7.0] and len(gh.issues) == 1
    assert "rate limit" in caplog.text


def test_primary_rate_limit_waits_until_reset():
    gh = FakeGitHub()
    gh.before = once(httpx.Response(403, headers={"x-ratelimit-remaining": "0", "x-ratelimit-reset": "1030"},
                                    json={"message": "API rate limit exceeded"}))
    sleeps = []
    gh.client(sleep=sleeps.append, clock=lambda: 1000.0).check_access()
    assert sleeps == [31.0]  # reset - now + 1s margin


def test_secondary_limit_without_headers_waits_a_minute():
    gh = FakeGitHub()
    gh.before = once(httpx.Response(429, json={"message": "slow down"}))
    sleeps = []
    gh.client(sleep=sleeps.append).check_access()
    assert sleeps == [SECONDARY_LIMIT_WAIT]


def test_very_long_rate_limit_fails_clearly_instead_of_hanging():
    gh = FakeGitHub()
    gh.before = lambda req: httpx.Response(403, headers={"retry-after": str(int(MAX_WAIT_SECONDS) + 1)},
                                           json={"message": "secondary rate limit"})
    sleeps = []
    with pytest.raises(GitHubError, match="rate limit"):
        gh.client(sleep=sleeps.append).check_access()
    assert sleeps == []


def test_server_errors_are_retried():
    gh = FakeGitHub()
    gh.before = once(httpx.Response(502, json={"message": "Bad Gateway"}))
    sleeps = []
    gh.client(sleep=sleeps.append).check_access()
    assert sleeps == [1]


# --- factory ---
def test_make_github_env(monkeypatch, caplog):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
    monkeypatch.setenv("GITHUB_REPO", REPO)
    with caplog.at_level(logging.WARNING):
        assert make_github() is None
    assert "GITHUB_TOKEN not set" in caplog.text

    monkeypatch.setenv("GITHUB_TOKEN", "your-token-here")
    assert make_github() is None

    monkeypatch.setenv("GITHUB_TOKEN", "t")
    monkeypatch.setenv("GITHUB_REPO", "not-a-repo")
    assert make_github() is None

    monkeypatch.delenv("GITHUB_REPO")
    monkeypatch.setenv("GITHUB_REPOSITORY", "a/b")  # what GitHub Actions sets
    c = make_github()
    assert isinstance(c, GitHubClient) and c.repo == "a/b" and c.owner == "a"
    c.close()
