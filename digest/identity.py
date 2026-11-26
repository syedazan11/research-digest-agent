"""Who this copy of the agent is, for the User-Agent sent to arXiv and GitHub.

arXiv and GitHub ask API clients to identify themselves. A copy made from the template must not
claim to be the original repo, so the URL comes from, in order:
  1. config.yaml `project.url` (run.py exports it as DIGEST_PROJECT_URL)
  2. GITHUB_REPOSITORY, which GitHub Actions sets to the copy's own owner/name
  3. the software's homepage
"""

from __future__ import annotations

import os

SOFTWARE_URL = "https://github.com/syedazan11/research-digest-agent"
VERSION = "0.2"


def project_url() -> str:
    explicit = os.getenv("DIGEST_PROJECT_URL", "").strip()
    if explicit:
        return explicit
    repo = os.getenv("GITHUB_REPOSITORY", "").strip()
    if repo:
        return f"https://github.com/{repo}"
    return SOFTWARE_URL


def user_agent() -> str:
    return f"research-digest-agent/{VERSION} (+{project_url()})"
