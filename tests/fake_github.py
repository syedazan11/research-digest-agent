"""An in-memory GitHub for tests, plugged into the real GitHubClient via httpx.MockTransport.

Only the endpoints the project uses. `before(request)` can return a Response to inject errors
(401, rate limits, ...); `crash_on_comment` simulates the process dying mid-delivery.
"""

from __future__ import annotations

import json
import re
from typing import Callable

import httpx

from digest.github import GitHubClient

REPO = "owner/repo"
TOKEN = "ghp_test_token_never_logged"


class Crash(Exception):
    """Stands in for the process dying (power cut, Ctrl+C, CI timeout)."""


class FakeGitHub:
    def __init__(self):
        self.issues: dict[int, dict] = {}
        self.comments: dict[int, dict] = {}
        self.reactions: dict[int, list[dict]] = {}
        self.labels: set[str] = set()
        self.log: list[tuple[str, str]] = []  # (method, path) of every request
        self.before: Callable[[httpx.Request], httpx.Response | None] | None = None
        # crash_on_comment = (n, "before"|"after"): die on the n-th comment POST, before or after it's saved
        self.crash_on_comment: tuple[int, str] | None = None
        self._comment_posts = 0
        self._next_issue = 1
        self._next_comment = 1000

    # --- helpers for tests ---
    def client(self, **kw) -> GitHubClient:
        kw.setdefault("sleep", lambda s: None)
        return GitHubClient(TOKEN, REPO, transport=httpx.MockTransport(self.handle), **kw)

    def react(self, comment_id: int, login: str, content: str, at: str = "2026-11-26T10:00:00Z") -> None:
        self.reactions.setdefault(comment_id, []).append(
            {"user": {"login": login}, "content": content, "created_at": at})

    def unreact(self, comment_id: int, login: str) -> None:
        self.reactions[comment_id] = [r for r in self.reactions.get(comment_id, []) if r["user"]["login"] != login]

    def comments_on(self, issue: int) -> list[dict]:
        return [c for c in self.comments.values() if c["issue"] == issue]

    def count(self, method: str, pattern: str) -> int:
        return sum(1 for m, p in self.log if m == method and re.fullmatch(pattern, p))

    # --- transport ---
    def handle(self, request: httpx.Request) -> httpx.Response:
        method, path = request.method, request.url.path
        self.log.append((method, path))
        if self.before is not None:
            injected = self.before(request)
            if injected is not None:
                return injected
        body = json.loads(request.content) if request.content else {}
        base = f"/repos/{REPO}"

        if method == "GET" and path == base:
            return httpx.Response(200, json={"full_name": REPO})
        if method == "POST" and path == f"{base}/labels":
            if body["name"] in self.labels:
                return httpx.Response(422, json={"message": "Validation Failed"})
            self.labels.add(body["name"])
            return httpx.Response(201, json=body)
        if method == "POST" and path == f"{base}/issues":
            n = self._next_issue
            self._next_issue += 1
            self.issues[n] = {"number": n, "title": body["title"], "body": body["body"], "state": "open",
                              "labels": [{"name": x} for x in body.get("labels", [])],
                              "html_url": f"https://github.com/{REPO}/issues/{n}"}
            return httpx.Response(201, json=self.issues[n])
        if method == "GET" and path == f"{base}/issues":
            label = request.url.params.get("labels")
            items = [i for i in self.issues.values() if not label or label in [x["name"] for x in i["labels"]]]
            return httpx.Response(200, json=self._page(request, items))

        m = re.fullmatch(rf"{base}/issues/(\d+)", path)
        if m and method == "PATCH":
            issue = self.issues.get(int(m.group(1)))
            if issue is None:
                return httpx.Response(404, json={"message": "Not Found"})
            issue.update({k: v for k, v in body.items() if k == "state"})
            return httpx.Response(200, json=issue)

        m = re.fullmatch(rf"{base}/issues/(\d+)/comments", path)
        if m:
            n = int(m.group(1))
            if n not in self.issues:
                return httpx.Response(404, json={"message": "Not Found"})
            if method == "GET":
                return httpx.Response(200, json=self._page(request, self.comments_on(n)))
            self._comment_posts += 1
            crash = self.crash_on_comment
            if crash and crash[0] == self._comment_posts and crash[1] == "before":
                raise Crash("died before GitHub saved the comment")
            cid = self._next_comment
            self._next_comment += 1
            self.comments[cid] = {"id": cid, "issue": n, "body": body["body"],
                                  "html_url": f"https://github.com/{REPO}/issues/{n}#issuecomment-{cid}"}
            if crash and crash[0] == self._comment_posts and crash[1] == "after":
                raise Crash("died after GitHub saved the comment, before we heard back")
            return httpx.Response(201, json=self.comments[cid])

        m = re.fullmatch(rf"{base}/issues/comments/(\d+)/reactions", path)
        if m and method == "GET":
            cid = int(m.group(1))
            if cid not in self.comments:
                return httpx.Response(404, json={"message": "Not Found"})
            return httpx.Response(200, json=self._page(request, self.reactions.get(cid, [])))

        return httpx.Response(404, json={"message": f"fake has no route for {method} {path}"})

    @staticmethod
    def _page(request: httpx.Request, items: list) -> list:
        per_page = int(request.url.params.get("per_page", 30))
        page = int(request.url.params.get("page", 1))
        return items[(page - 1) * per_page : page * per_page]
