"""GitHub client for the reviewer agent.

Thin async wrapper over ``githubkit`` — the single entry point for
every GitHub API call the reviewer makes. Keeping it reviewer-local
(rather than importing developer's client) matches the FR's
functional-boundary rule: reviewer owns its own review posting, and
the only cross-agent link to developer is through bus events + review
ingestion on the consumer side.

Token discovery order:
    1. explicit ``token`` kwarg
    2. ``reviewer.credentials.get_github_token()`` chain
       (``GITHUB_TOKEN`` -> ``GH_TOKEN`` -> ``gh auth token``)
    3. unauthenticated read (public repos only; review posting will 401)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class GithubClientError(RuntimeError):
    """Base class for reviewer GitHub client failures."""


class GithubNotFoundError(GithubClientError):
    """Raised when a repo or PR does not exist (or the caller can't see it)."""


class GithubAuthError(GithubClientError):
    """Raised when GitHub rejects the supplied credentials."""


@dataclass
class PRMetadata:
    """The slice of PR state the reviewer actually consults."""

    repo: str
    number: int
    title: str = ""
    body: str = ""
    state: str = ""
    is_draft: bool = False
    base_ref: str = ""
    base_sha: str = ""
    head_ref: str = ""
    head_sha: str = ""
    labels: list[str] = field(default_factory=list)
    author: str = ""
    html_url: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo": self.repo,
            "number": self.number,
            "title": self.title,
            "body": self.body,
            "state": self.state,
            "is_draft": self.is_draft,
            "base_ref": self.base_ref,
            "base_sha": self.base_sha,
            "head_ref": self.head_ref,
            "head_sha": self.head_sha,
            "labels": list(self.labels),
            "author": self.author,
            "html_url": self.html_url,
        }


@dataclass
class SubmittedReview:
    """The subset of GitHub's review response the reviewer returns to callers."""

    id: int
    html_url: str
    body: str
    state: str  # "COMMENTED" | "APPROVED" | "CHANGES_REQUESTED" | "PENDING"
    submitted_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "html_url": self.html_url,
            "body": self.body,
            "state": self.state,
            "submitted_at": self.submitted_at,
        }


class ReviewerGithubClient:
    """Async thin wrapper over githubkit for the reviewer.

    Lazy-constructs the underlying ``GitHub`` client on first use so
    importing this module has zero cost when the reviewer never reaches
    for GitHub (unit tests, CLI-only paths). Tests inject a fake
    ``gh_client`` to avoid the real githubkit surface entirely.
    """

    def __init__(
        self,
        token: str | None = None,
        *,
        gh_client: Any | None = None,
    ):
        # Credential discovery lives in reviewer.credentials so this
        # repo never grows its own keyring / file-reading logic. An
        # explicit ``token`` kwarg still wins; otherwise we ask the
        # credentials module which in turn chains env vars + gh's CLI.
        from reviewer.credentials import get_github_token

        self._token = token or get_github_token()
        self._gh: Any = gh_client

    def _client(self) -> Any:
        if self._gh is None:
            try:
                from githubkit import GitHub
            except ImportError as exc:
                raise GithubClientError(
                    "githubkit is not installed; add it to pyproject.toml "
                    "or remove the GitHub dependency from the reviewer "
                    "agent."
                ) from exc
            # Keyword form avoids reliance on the positional-arg order
            # of githubkit's GitHub constructor across minor versions.
            # As of githubkit ~0.13, the constructor takes ``auth`` —
            # accepting either a string token or a TokenAuth wrapper —
            # rather than the older ``token`` kwarg. Passing a bare
            # string is still supported. Older versions used ``token``;
            # operators on a stale install will see TypeError here and
            # the bus skill (review_pr) will surface
            # ``error_category="backend_error"`` with the kwarg name in
            # the message, which is the diagnostic operators need.
            self._gh = GitHub(auth=self._token) if self._token else GitHub()
        return self._gh

    # -- reads ---------------------------------------------------------

    async def get_pr_metadata(self, repo: str, pr_number: int) -> PRMetadata:
        """Return normalized PR metadata. Raises :class:`GithubNotFoundError`
        if the repo/PR doesn't exist (or isn't visible to the token)."""
        owner, name = _split_repo(repo)
        try:
            resp = await self._client().rest.pulls.async_get(
                owner, name, int(pr_number)
            )
        except Exception as exc:  # githubkit raises various types
            raise _classify(exc, f"get_pr_metadata({repo}#{pr_number})") from exc

        pr = resp.parsed_data
        return PRMetadata(
            repo=repo,
            number=int(pr_number),
            title=getattr(pr, "title", "") or "",
            body=getattr(pr, "body", "") or "",
            state=getattr(pr, "state", "") or "",
            is_draft=bool(getattr(pr, "draft", False)),
            base_ref=_attr(pr, "base", "ref", default=""),
            base_sha=_attr(pr, "base", "sha", default=""),
            head_ref=_attr(pr, "head", "ref", default=""),
            head_sha=_attr(pr, "head", "sha", default=""),
            labels=[
                getattr(label, "name", "")
                for label in (getattr(pr, "labels", None) or [])
                if getattr(label, "name", "")
            ],
            author=_attr(pr, "user", "login", default=""),
            html_url=getattr(pr, "html_url", "") or "",
        )

    async def get_pr_diff(self, repo: str, pr_number: int) -> str:
        """Return the unified diff for the PR.

        Uses the GitHub media-type trick: same pulls endpoint, different
        ``Accept`` header, bytes response is raw diff text.
        """
        owner, name = _split_repo(repo)
        try:
            resp = await self._client().arequest(
                "GET",
                f"/repos/{owner}/{name}/pulls/{int(pr_number)}",
                headers={"Accept": "application/vnd.github.v3.diff"},
            )
        except Exception as exc:
            raise _classify(exc, f"get_pr_diff({repo}#{pr_number})") from exc

        # Response body is the raw diff; most githubkit responses expose
        # either .content (bytes) or .text.
        body = _response_text(resp)
        if body is None:
            raise GithubClientError(
                f"get_pr_diff({repo}#{pr_number}): no diff content in response"
            )
        return body

    # -- writes --------------------------------------------------------

    async def submit_review(
        self,
        repo: str,
        pr_number: int,
        *,
        body: str,
        comments: list[dict[str, Any]],
        event: str = "COMMENT",
        commit_sha: str | None = None,
    ) -> SubmittedReview:
        """Post a pull-request review. ``event`` is one of GitHub's set:
        ``COMMENT``, ``APPROVE``, ``REQUEST_CHANGES``, ``PENDING``.

        ``comments`` is the raw GitHub inline-comment shape —
        ``{"path": str, "line": int, "body": str, "side": "RIGHT"}``.
        The skill layer is responsible for building this list from
        :class:`ReviewFinding` records; this method stays dumb about
        review-lib types.
        """
        owner, name = _split_repo(repo)
        payload: dict[str, Any] = {
            "body": body,
            "event": event,
            "comments": list(comments),
        }
        if commit_sha:
            payload["commit_id"] = commit_sha
        try:
            resp = await self._client().rest.pulls.async_create_review(
                owner, name, int(pr_number), **payload
            )
        except Exception as exc:
            raise _classify(exc, f"submit_review({repo}#{pr_number})") from exc

        review = resp.parsed_data
        return SubmittedReview(
            id=int(getattr(review, "id", 0) or 0),
            html_url=getattr(review, "html_url", "") or "",
            body=getattr(review, "body", "") or body,
            state=getattr(review, "state", "") or event,
            submitted_at=str(getattr(review, "submitted_at", "") or ""),
        )


def _split_repo(repo: str) -> tuple[str, str]:
    """Split ``"owner/name"`` into ``(owner, name)``. Raises on malformed input."""
    if "/" not in repo or repo.count("/") != 1:
        raise GithubClientError(
            f"repo must be in 'owner/name' form, got {repo!r}"
        )
    owner, name = repo.split("/", 1)
    if not owner or not name:
        raise GithubClientError(
            f"repo must be in 'owner/name' form, got {repo!r}"
        )
    return owner, name


def _attr(obj: Any, *path: str, default: Any = None) -> Any:
    """Walk ``path`` as a sequence of attribute lookups, returning ``default``
    as soon as any step is ``None`` or absent."""
    cur = obj
    for step in path:
        cur = getattr(cur, step, None)
        if cur is None:
            return default
    return cur


def _response_text(resp: Any) -> str | None:
    """Pull text out of a githubkit response regardless of its shape."""
    content = getattr(resp, "content", None)
    if isinstance(content, (bytes, bytearray)):
        return content.decode("utf-8", errors="replace")
    text = getattr(resp, "text", None)
    if isinstance(text, str):
        return text
    if isinstance(content, str):
        return content
    return None


def _classify(exc: Exception, context: str) -> GithubClientError:
    """Map githubkit / HTTP errors into the reviewer's error taxonomy."""
    status = getattr(exc, "status_code", None) or getattr(
        getattr(exc, "response", None), "status_code", None
    )
    msg = f"{context}: {exc}"
    if status == 401 or status == 403:
        return GithubAuthError(msg)
    if status == 404:
        return GithubNotFoundError(msg)
    return GithubClientError(msg)


__all__ = [
    "GithubAuthError",
    "GithubClientError",
    "GithubNotFoundError",
    "PRMetadata",
    "ReviewerGithubClient",
    "SubmittedReview",
]
