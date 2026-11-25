"""A small GitHub REST client: just the calls delivery and feedback need.

Why not PyGithub? Six endpoints don't justify a dependency, and owning the request loop means
the rate-limit handling is explicit, logged, and tested.

Error policy:
- rate limited (primary or secondary): wait for retry-after / x-ratelimit-reset, then retry.
- 401 / 403 (not a rate limit): GitHubAuthError with a clear message. Retrying can't fix a token.
- 5xx / network: short backoff and retry.
The token never appears in logs or error messages.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Callable

import httpx

log = logging.getLogger(__name__)

API = "https://api.github.com"
USER_AGENT = "research-digest-agent/0.1 (+https://github.com/syedazan11/research-digest-agent)"
PLACEHOLDER_TOKEN = "your-token-here"
SECONDARY_LIMIT_WAIT = 60.0  # GitHub's docs: with no retry-after, wait at least a minute
MAX_WAIT_SECONDS = 120.0  # never sleep longer than this on one rate limit; fail clearly instead


class GitHubError(RuntimeError):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class GitHubAuthError(GitHubError):
    """Bad/expired token, missing permission, or a repo the token can't see. Stop, don't retry."""


class GitHubNotFound(GitHubError):
    """A specific object (e.g. a deleted comment) is gone. Callers usually skip it."""


def _message(r: httpx.Response) -> str:
    try:
        return str(r.json().get("message", ""))[:200]
    except Exception:
        return r.text[:200]


class GitHubClient:
    def __init__(
        self,
        token: str,
        repo: str,
        transport: httpx.BaseTransport | None = None,  # tests pass httpx.MockTransport
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.time,
        max_retries: int = 3,
    ):
        if repo.count("/") != 1:
            raise ValueError(f"GITHUB_REPO must look like owner/name, got {repo!r}")
        self.repo = repo
        self.owner = repo.split("/")[0]
        self.sleep = sleep
        self.clock = clock
        self.max_retries = max_retries
        self.requests = 0
        self.http = httpx.Client(
            base_url=API,
            transport=transport,
            timeout=30.0,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": USER_AGENT,
            },
        )

    def close(self) -> None:
        self.http.close()

    # --- request loop ---
    def _rate_limit_wait(self, r: httpx.Response) -> float | None:
        """Seconds to wait if this response is a rate limit, else None."""
        if r.status_code not in (403, 429):
            return None
        retry_after = r.headers.get("retry-after")
        if retry_after:
            try:
                return float(retry_after)
            except ValueError:
                return SECONDARY_LIMIT_WAIT
        if r.headers.get("x-ratelimit-remaining") == "0" and r.headers.get("x-ratelimit-reset"):
            try:
                return max(0.0, float(r.headers["x-ratelimit-reset"]) - self.clock()) + 1.0
            except ValueError:
                return SECONDARY_LIMIT_WAIT
        if r.status_code == 429 or "rate limit" in _message(r).lower():
            return SECONDARY_LIMIT_WAIT
        return None

    def request(self, method: str, path: str, **kwargs) -> httpx.Response:
        for attempt in range(self.max_retries + 1):
            last = attempt == self.max_retries
            self.requests += 1
            try:
                r = self.http.request(method, path, **kwargs)
            except httpx.HTTPError as e:
                if last:
                    raise GitHubError(f"GitHub unreachable ({type(e).__name__}) on {method} {path}") from e
                log.warning("GitHub network error (%s), retrying in %ds", type(e).__name__, 2**attempt)
                self.sleep(2**attempt)
                continue

            if r.status_code < 400:
                return r

            wait = self._rate_limit_wait(r)
            if wait is not None:
                if last or wait > MAX_WAIT_SECONDS:
                    raise GitHubError(f"GitHub rate limit on {method} {path}: retry in {wait:.0f}s", r.status_code)
                log.warning("GitHub rate limit (%d), waiting %.0fs", r.status_code, wait)
                self.sleep(wait)
                continue
            if r.status_code == 401:
                raise GitHubAuthError("GITHUB_TOKEN is invalid or expired (401): make a new token", 401)
            if r.status_code == 403:
                raise GitHubAuthError(
                    f"GITHUB_TOKEN lacks permission on {self.repo} (403: {_message(r)}). "
                    "It needs Issues: Read and write on this repo.", 403)
            if r.status_code == 404:
                raise GitHubNotFound(f"GitHub 404 on {method} {path}", 404)
            if r.status_code >= 500 and not last:
                log.warning("GitHub %d, retrying in %ds", r.status_code, 2**attempt)
                self.sleep(2**attempt)
                continue
            raise GitHubError(f"GitHub {r.status_code} on {method} {path}: {_message(r)}", r.status_code)
        raise GitHubError("unreachable")  # loop always returns or raises

    def _paginate(self, path: str, params: dict | None = None, max_pages: int = 10) -> list[dict]:
        out: list[dict] = []
        params = {**(params or {}), "per_page": 100}
        for page in range(1, max_pages + 1):
            batch = self.request("GET", path, params={**params, "page": page}).json()
            out.extend(batch)
            if len(batch) < 100:
                break
        return out

    # --- endpoints ---
    def check_access(self) -> None:
        """Fail fast with a clear message if the token or repo is wrong."""
        try:
            self.request("GET", f"/repos/{self.repo}")
        except GitHubNotFound:
            raise GitHubAuthError(
                f"Repo {self.repo} not found, or GITHUB_TOKEN can't access it (check GITHUB_REPO and the "
                "token's repository access)", 404) from None

    def ensure_label(self, name: str, color: str = "0e8a16", description: str = "") -> None:
        try:
            self.request("POST", f"/repos/{self.repo}/labels",
                         json={"name": name, "color": color, "description": description})
        except GitHubError as e:
            if e.status != 422:  # 422 = label already exists
                raise

    def create_issue(self, title: str, body: str, labels: list[str]) -> dict:
        return self.request("POST", f"/repos/{self.repo}/issues",
                            json={"title": title, "body": body, "labels": labels}).json()

    def close_issue(self, number: int) -> None:
        self.request("PATCH", f"/repos/{self.repo}/issues/{number}",
                     json={"state": "closed", "state_reason": "completed"})

    def list_issues(self, label: str, state: str = "all") -> list[dict]:
        issues = self._paginate(f"/repos/{self.repo}/issues", {"labels": label, "state": state}, max_pages=3)
        return [i for i in issues if "pull_request" not in i]  # this endpoint also returns PRs

    def create_comment(self, issue_number: int, body: str) -> dict:
        return self.request("POST", f"/repos/{self.repo}/issues/{issue_number}/comments",
                            json={"body": body}).json()

    def list_comments(self, issue_number: int) -> list[dict]:
        return self._paginate(f"/repos/{self.repo}/issues/{issue_number}/comments")

    def list_reactions(self, comment_id: int) -> list[dict]:
        return self._paginate(f"/repos/{self.repo}/issues/comments/{comment_id}/reactions")


def make_github() -> GitHubClient | None:
    """Client from GITHUB_TOKEN + GITHUB_REPO (or GITHUB_REPOSITORY, set by GitHub Actions); None with a warning."""
    token = os.getenv("GITHUB_TOKEN", "").strip()
    repo = (os.getenv("GITHUB_REPO", "") or os.getenv("GITHUB_REPOSITORY", "")).strip()
    if not token or token == PLACEHOLDER_TOKEN:
        log.warning("GITHUB_TOKEN not set: skipping GitHub delivery (see .env.example)")
        return None
    if not repo:
        log.warning("GITHUB_REPO not set (owner/name): skipping GitHub delivery")
        return None
    try:
        return GitHubClient(token, repo)
    except ValueError as e:
        log.warning("%s: skipping GitHub delivery", e)
        return None
