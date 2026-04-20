"""Tests for the pure helpers + injection surface of ``ReviewerGithubClient``.

Exercising the full githubkit round-trip requires a real GitHub instance
or a heavy mock of its internals; that's out of scope for this unit
test. Here we cover the small pure helpers (``_split_repo``,
``_classify``, ``_response_text``) and verify that the ``gh_client=``
injection point lets tests swap in a fake that the client calls into.
"""

from __future__ import annotations

from typing import Any

import pytest

from reviewer.github_client import (
    GithubAuthError,
    GithubClientError,
    GithubNotFoundError,
    ReviewerGithubClient,
    _classify,
    _response_text,
    _split_repo,
)


# ---------------------------------------------------------------------------
# _split_repo
# ---------------------------------------------------------------------------


def test_split_repo_happy_path():
    assert _split_repo("tolldog/khonliang-reviewer") == ("tolldog", "khonliang-reviewer")


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "no-slash",
        "/missing-owner",
        "missing-name/",
        "too/many/parts",
    ],
)
def test_split_repo_rejects_malformed(bad: str):
    with pytest.raises(GithubClientError):
        _split_repo(bad)


# ---------------------------------------------------------------------------
# _classify
# ---------------------------------------------------------------------------


class _FakeHTTPError(Exception):
    def __init__(self, status_code: int, message: str = "boom"):
        super().__init__(message)
        self.status_code = status_code


def test_classify_404_is_not_found():
    err = _classify(_FakeHTTPError(404), "get_pr_metadata(a/b#1)")
    assert isinstance(err, GithubNotFoundError)


def test_classify_401_is_auth_error():
    err = _classify(_FakeHTTPError(401), "submit_review(a/b#1)")
    assert isinstance(err, GithubAuthError)


def test_classify_403_is_auth_error():
    err = _classify(_FakeHTTPError(403), "submit_review(a/b#1)")
    assert isinstance(err, GithubAuthError)


def test_classify_other_status_falls_through_to_base():
    err = _classify(_FakeHTTPError(500), "ctx")
    assert type(err) is GithubClientError
    assert not isinstance(err, (GithubAuthError, GithubNotFoundError))


def test_classify_no_status_falls_through_to_base():
    err = _classify(RuntimeError("unexpected"), "ctx")
    assert type(err) is GithubClientError


def test_classify_preserves_context_in_message():
    err = _classify(_FakeHTTPError(404, "Not Found"), "get_pr_diff(x/y#99)")
    assert "get_pr_diff(x/y#99)" in str(err)
    assert "Not Found" in str(err)


# ---------------------------------------------------------------------------
# _response_text
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, *, content: Any = None, text: Any = None):
        if content is not None:
            self.content = content
        if text is not None:
            self.text = text


def test_response_text_from_bytes_content():
    resp = _FakeResponse(content=b"diff --git\n-a\n+b\n")
    assert _response_text(resp) == "diff --git\n-a\n+b\n"


def test_response_text_from_str_text():
    resp = _FakeResponse(text="diff --git")
    assert _response_text(resp) == "diff --git"


def test_response_text_from_str_content():
    resp = _FakeResponse(content="diff --git")
    assert _response_text(resp) == "diff --git"


def test_response_text_none_when_no_readable_field():
    class _Bare:
        pass

    assert _response_text(_Bare()) is None


# ---------------------------------------------------------------------------
# Injection surface (async)
# ---------------------------------------------------------------------------


async def test_client_routes_through_injected_gh():
    """A gh_client passed in is used as-is; token + lazy init are skipped."""
    calls: list[tuple[str, Any, Any, Any]] = []

    class _FakePR:
        title = "PR title"
        body = "PR body"
        state = "open"
        draft = False
        html_url = "https://gh/x/y/pull/1"
        user = type("U", (), {"login": "tolldog"})
        base = type("B", (), {"ref": "main", "sha": "base-sha"})
        head = type("H", (), {"ref": "feat/x", "sha": "head-sha"})
        labels = []

    class _FakeResp:
        parsed_data = _FakePR()

    class _FakePulls:
        async def async_get(self, owner, name, pr_number):
            calls.append(("async_get", owner, name, pr_number))
            return _FakeResp()

    class _FakeRest:
        pulls = _FakePulls()

    class _FakeGH:
        rest = _FakeRest()

    client = ReviewerGithubClient(gh_client=_FakeGH())
    meta = await client.get_pr_metadata("tolldog/example", 1)

    assert calls == [("async_get", "tolldog", "example", 1)]
    assert meta.title == "PR title"
    assert meta.head_sha == "head-sha"
    assert meta.base_ref == "main"
    assert meta.author == "tolldog"


async def test_client_classifies_fetch_failures():
    class _FakePulls:
        async def async_get(self, owner, name, pr_number):
            raise _FakeHTTPError(404, "Not Found")

    class _FakeRest:
        pulls = _FakePulls()

    class _FakeGH:
        rest = _FakeRest()

    client = ReviewerGithubClient(gh_client=_FakeGH())
    with pytest.raises(GithubNotFoundError):
        await client.get_pr_metadata("tolldog/example", 999)


async def test_submit_review_forwards_payload():
    class _FakeReview:
        id = 123
        html_url = "https://gh/r/123"
        body = "posted body"
        state = "COMMENTED"
        submitted_at = "2026-04-20T10:00:00Z"

    class _FakeResp:
        parsed_data = _FakeReview()

    captured: dict[str, Any] = {}

    class _FakePulls:
        async def async_create_review(self, owner, name, pr_number, **kwargs):
            captured["owner"] = owner
            captured["name"] = name
            captured["pr_number"] = pr_number
            captured["kwargs"] = kwargs
            return _FakeResp()

    class _FakeRest:
        pulls = _FakePulls()

    class _FakeGH:
        rest = _FakeRest()

    client = ReviewerGithubClient(gh_client=_FakeGH())
    submitted = await client.submit_review(
        "tolldog/example",
        7,
        body="body",
        comments=[{"path": "a", "line": 1, "body": "b", "side": "RIGHT"}],
        event="COMMENT",
        commit_sha="deadbeef",
    )

    assert captured["owner"] == "tolldog"
    assert captured["name"] == "example"
    assert captured["pr_number"] == 7
    assert captured["kwargs"]["body"] == "body"
    assert captured["kwargs"]["event"] == "COMMENT"
    assert captured["kwargs"]["commit_id"] == "deadbeef"
    assert len(captured["kwargs"]["comments"]) == 1
    assert submitted.id == 123
    assert submitted.state == "COMMENTED"
