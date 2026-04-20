"""Tests for :meth:`ReviewerAgent.handle_review_pr` + the GitHub formatter."""

from __future__ import annotations

from typing import Any

from khonliang_bus.testing import AgentTestHarness

from khonliang_reviewer import (
    ReviewFinding,
    ReviewProvider,
    ReviewRequest,
    ReviewResult,
    UsageEvent,
)
from reviewer.agent import ReviewerAgent, _format_for_github
from reviewer.github_client import (
    GithubClientError,
    GithubNotFoundError,
    PRMetadata,
    SubmittedReview,
)
from reviewer.selector import ProviderSelector, SelectorConfig


# ---------------------------------------------------------------------------
# _format_for_github (pure)
# ---------------------------------------------------------------------------


def test_format_empty_result_returns_placeholder_body():
    body, comments = _format_for_github({"summary": "", "findings": []})
    assert body == "No findings."
    assert comments == []


def test_format_summary_only_result():
    body, comments = _format_for_github(
        {"summary": "Looks fine.", "findings": []}
    )
    assert body == "Looks fine."
    assert comments == []


def test_format_anchored_finding_produces_inline_comment():
    body, comments = _format_for_github(
        {
            "summary": "LGTM with one nit.",
            "findings": [
                {
                    "severity": "nit",
                    "title": "Variable name",
                    "body": "`paths` mis-names a str return.",
                    "path": "README.md",
                    "line": 66,
                    "suggestion": "tree = trace_chain(...)",
                }
            ],
        }
    )
    assert body == "LGTM with one nit."
    assert len(comments) == 1
    comment = comments[0]
    assert comment["path"] == "README.md"
    assert comment["line"] == 66
    assert comment["side"] == "RIGHT"
    assert "🟢 Nit" in comment["body"]
    assert "Variable name" in comment["body"]
    assert "`paths` mis-names" in comment["body"]
    assert "```suggestion" in comment["body"]
    assert "tree = trace_chain" in comment["body"]


def test_format_summary_level_finding_appended_to_body():
    body, comments = _format_for_github(
        {
            "summary": "Overall review.",
            "findings": [
                {
                    "severity": "comment",
                    "title": "General observation",
                    "body": "Consider consolidating duplicate code.",
                    "path": None,
                    "line": None,
                }
            ],
        }
    )
    assert "Overall review." in body
    assert "Additional notes" in body
    assert "🟡 Comment" in body
    assert "General observation" in body
    assert "duplicate code" in body
    assert comments == []


def test_format_zero_line_finding_falls_back_to_summary_level():
    """GitHub rejects line=0; the formatter must demote to summary-level."""
    body, comments = _format_for_github(
        {
            "summary": "",
            "findings": [
                {
                    "severity": "nit",
                    "title": "Zero line",
                    "body": "bad line number",
                    "path": "a.py",
                    "line": 0,
                }
            ],
        }
    )
    assert comments == []
    assert "Zero line" in body


def test_format_negative_line_finding_falls_back_to_summary_level():
    """Similarly, negative line numbers can't anchor — demote, don't post."""
    body, comments = _format_for_github(
        {
            "summary": "",
            "findings": [
                {
                    "severity": "nit",
                    "title": "Negative line",
                    "body": "x",
                    "path": "a.py",
                    "line": -3,
                }
            ],
        }
    )
    assert comments == []
    assert "Negative line" in body


def test_format_bool_line_finding_falls_back_to_summary_level():
    """`bool` subclasses `int`; `True`/`False` must NOT pass as line numbers."""
    for val in (True, False):
        body, comments = _format_for_github(
            {
                "summary": "",
                "findings": [
                    {
                        "severity": "nit",
                        "title": "Bool line",
                        "body": "x",
                        "path": "a.py",
                        "line": val,
                    }
                ],
            }
        )
        assert comments == []
        assert "Bool line" in body


def test_format_additional_notes_heading_even_with_empty_summary():
    """Summary-level findings must carry the heading even without a top summary."""
    body, comments = _format_for_github(
        {
            "summary": "",
            "findings": [
                {
                    "severity": "nit",
                    "title": "Style",
                    "body": "Prefer trailing commas.",
                    "path": None,
                    "line": None,
                }
            ],
        }
    )
    assert body.startswith("### Additional notes")
    assert "🟢 Nit" in body
    assert comments == []


def test_format_mixed_anchored_and_summary_level_findings():
    body, comments = _format_for_github(
        {
            "summary": "Two findings.",
            "findings": [
                {
                    "severity": "concern",
                    "title": "Missing tests",
                    "body": "Empty-input path uncovered.",
                    "path": "pkg/mod.py",
                    "line": 42,
                },
                {
                    "severity": "nit",
                    "title": "Style",
                    "body": "Prefer trailing commas.",
                    "path": None,
                    "line": None,
                },
            ],
        }
    )
    assert len(comments) == 1
    assert comments[0]["path"] == "pkg/mod.py"
    assert "Additional notes" in body
    assert "🟢 Nit" in body
    assert "🔴 Concern" not in body  # concern is inline, not in body
    assert comments[0]["body"].startswith("**🔴 Concern")


def test_format_skips_non_dict_finding_items():
    body, comments = _format_for_github(
        {
            "summary": "ok",
            "findings": [
                "rogue string",
                None,
                42,
                {
                    "severity": "nit",
                    "title": "valid",
                    "body": "x",
                    "path": "a.py",
                    "line": 1,
                },
            ],
        }
    )
    assert len(comments) == 1
    assert comments[0]["body"].startswith("**🟢 Nit — valid")


def test_format_tolerates_non_list_findings():
    body, comments = _format_for_github({"summary": "s", "findings": "not-a-list"})
    assert body == "s"
    assert comments == []


# ---------------------------------------------------------------------------
# Fakes for review_pr tests
# ---------------------------------------------------------------------------


class _RecordingProvider(ReviewProvider):
    def __init__(self, name: str, response: ReviewResult):
        self.name = name
        self._response = response
        self.last_request: ReviewRequest | None = None

    async def review(self, request: ReviewRequest) -> ReviewResult:
        self.last_request = request
        return self._response


def _make_result(
    *,
    summary: str = "ok",
    findings: list[ReviewFinding] | None = None,
) -> ReviewResult:
    return ReviewResult(
        request_id="req-pr",
        summary=summary,
        findings=findings or [],
        disposition="posted",
        backend="ollama",
        model="qwen3.5",
        usage=UsageEvent(
            timestamp=1.0,
            backend="ollama",
            model="qwen3.5",
            input_tokens=50,
            output_tokens=20,
        ),
    )


class _FakeGithub:
    """Stand-in for :class:`ReviewerGithubClient` — records calls + returns canned data."""

    def __init__(
        self,
        *,
        metadata: PRMetadata | None = None,
        diff: str = "diff --git a/f b/f\n@@ -1 +1 @@\n-old\n+new\n",
        submit_result: SubmittedReview | None = None,
        metadata_error: Exception | None = None,
        diff_error: Exception | None = None,
        submit_error: Exception | None = None,
    ):
        self._metadata = metadata or PRMetadata(
            repo="tolldog/example",
            number=42,
            title="Test PR",
            head_sha="abc123",
            head_ref="feat/x",
            base_ref="main",
        )
        self._diff = diff
        self._submit_result = submit_result or SubmittedReview(
            id=999, html_url="https://gh/tolldog/example/pull/42#pullrequestreview-999",
            body="", state="COMMENTED",
        )
        self._metadata_error = metadata_error
        self._diff_error = diff_error
        self._submit_error = submit_error
        self.submit_calls: list[dict[str, Any]] = []

    async def get_pr_metadata(self, repo: str, pr_number: int) -> PRMetadata:
        if self._metadata_error is not None:
            raise self._metadata_error
        return self._metadata

    async def get_pr_diff(self, repo: str, pr_number: int) -> str:
        if self._diff_error is not None:
            raise self._diff_error
        return self._diff

    async def submit_review(
        self, repo: str, pr_number: int, *, body: str, comments, event, commit_sha=None
    ) -> SubmittedReview:
        if self._submit_error is not None:
            raise self._submit_error
        self.submit_calls.append(
            {
                "repo": repo,
                "pr_number": pr_number,
                "body": body,
                "comments": list(comments),
                "event": event,
                "commit_sha": commit_sha,
            }
        )
        return self._submit_result


def _make_harness(
    provider: _RecordingProvider,
    *,
    github: _FakeGithub | None = None,
) -> AgentTestHarness:
    selector = ProviderSelector(
        {"ollama": provider},
        SelectorConfig(default_backend="ollama", default_model="qwen3.5"),
    )
    return AgentTestHarness(
        ReviewerAgent,
        selector=selector,
        github_client=github or _FakeGithub(),
    )


# ---------------------------------------------------------------------------
# review_pr happy paths
# ---------------------------------------------------------------------------


async def test_review_pr_fetches_reviews_and_posts():
    finding = ReviewFinding(
        severity="nit",
        title="Typo",
        body="'recieve' -> 'receive'",
        path="README.md",
        line=10,
    )
    provider = _RecordingProvider("ollama", _make_result(summary="one finding", findings=[finding]))
    github = _FakeGithub()
    harness = _make_harness(provider, github=github)

    result = await harness.call(
        "review_pr", {"repo": "tolldog/example", "pr_number": 42}
    )

    # provider saw the diff
    assert provider.last_request is not None
    assert "diff --git" in provider.last_request.content
    assert provider.last_request.kind == "pr_diff"
    assert provider.last_request.context["pr"]["number"] == 42
    # github submit was called with the formatted body + comments
    assert len(github.submit_calls) == 1
    call = github.submit_calls[0]
    assert call["repo"] == "tolldog/example"
    assert call["pr_number"] == 42
    assert call["event"] == "COMMENT"
    assert call["commit_sha"] == "abc123"
    assert len(call["comments"]) == 1
    assert call["comments"][0]["path"] == "README.md"
    # result includes the github section + pr metadata
    assert result["github"]["dry_run"] is False
    assert result["github"]["inline_comments_posted"] == 1
    assert result["github"]["review"]["id"] == 999
    assert result["pr"]["title"] == "Test PR"


async def test_review_pr_dry_run_does_not_post():
    provider = _RecordingProvider("ollama", _make_result())
    github = _FakeGithub()
    harness = _make_harness(provider, github=github)

    result = await harness.call(
        "review_pr",
        {"repo": "tolldog/example", "pr_number": 42, "dry_run": True},
    )

    assert github.submit_calls == []
    assert result["github"]["dry_run"] is True
    assert "body" in result["github"]
    assert "comments" in result["github"]


async def test_review_pr_honors_caller_backend_and_model_override():
    provider = _RecordingProvider("ollama", _make_result())
    github = _FakeGithub()
    harness = _make_harness(provider, github=github)

    await harness.call(
        "review_pr",
        {
            "repo": "tolldog/example",
            "pr_number": 42,
            "backend": "ollama",
            "model": "kimi-k2.5:cloud",
        },
    )

    assert provider.last_request is not None
    assert provider.last_request.metadata["model"] == "kimi-k2.5:cloud"


async def test_review_pr_threads_pr_context_into_review_request():
    provider = _RecordingProvider("ollama", _make_result())
    github = _FakeGithub(
        metadata=PRMetadata(
            repo="tolldog/example",
            number=42,
            title="Refactor fetcher",
            labels=["refactor", "ready-to-review"],
            head_sha="deadbeef",
            head_ref="feat/refactor",
            base_ref="main",
            author="tolldog",
        )
    )
    harness = _make_harness(provider, github=github)

    await harness.call("review_pr", {"repo": "tolldog/example", "pr_number": 42})

    ctx = provider.last_request.context
    pr = ctx["pr"]
    assert pr["title"] == "Refactor fetcher"
    assert pr["labels"] == ["refactor", "ready-to-review"]
    assert pr["head_sha"] == "deadbeef"
    assert pr["author"] == "tolldog"


async def test_review_pr_custom_event_flag_threads_through():
    provider = _RecordingProvider("ollama", _make_result())
    github = _FakeGithub()
    harness = _make_harness(provider, github=github)

    await harness.call(
        "review_pr",
        {"repo": "tolldog/example", "pr_number": 42, "event": "REQUEST_CHANGES"},
    )

    assert github.submit_calls[0]["event"] == "REQUEST_CHANGES"


async def test_review_pr_event_is_normalized_to_upper_case():
    provider = _RecordingProvider("ollama", _make_result())
    github = _FakeGithub()
    harness = _make_harness(provider, github=github)

    await harness.call(
        "review_pr",
        {"repo": "tolldog/example", "pr_number": 42, "event": "comment"},
    )

    assert github.submit_calls[0]["event"] == "COMMENT"


async def test_review_pr_rejects_unsupported_event():
    """Typos + unsupported events fail with a structured bus error, not a 4xx."""
    provider = _RecordingProvider("ollama", _make_result())
    github = _FakeGithub()
    harness = _make_harness(provider, github=github)

    result = await harness.call(
        "review_pr",
        {"repo": "tolldog/example", "pr_number": 42, "event": "SHIPIT"},
    )

    assert "error" in result
    assert "event must be one of" in result["error"]
    assert "SHIPIT" in result["error"]
    # GitHub surface never touched
    assert github.submit_calls == []
    # Provider also not invoked (validation happens before the review)
    assert provider.last_request is None


async def test_review_pr_rejects_approve_event():
    """Approval authority is human-only per FR; the skill must refuse APPROVE."""
    provider = _RecordingProvider("ollama", _make_result())
    github = _FakeGithub()
    harness = _make_harness(provider, github=github)

    result = await harness.call(
        "review_pr",
        {"repo": "tolldog/example", "pr_number": 42, "event": "APPROVE"},
    )

    assert "error" in result
    assert "APPROVE" in result["error"]
    # Provider + GitHub never touched
    assert github.submit_calls == []
    assert provider.last_request is None


# ---------------------------------------------------------------------------
# review_pr error paths
# ---------------------------------------------------------------------------


async def test_review_pr_missing_repo_returns_error():
    provider = _RecordingProvider("ollama", _make_result())
    harness = _make_harness(provider)
    result = await harness.call("review_pr", {"pr_number": 42})
    assert "error" in result
    assert "repo" in result["error"]


async def test_review_pr_missing_pr_number_returns_error():
    provider = _RecordingProvider("ollama", _make_result())
    harness = _make_harness(provider)
    result = await harness.call("review_pr", {"repo": "tolldog/example"})
    assert "error" in result
    assert "pr_number" in result["error"]


async def test_review_pr_non_integer_pr_number_returns_error():
    provider = _RecordingProvider("ollama", _make_result())
    harness = _make_harness(provider)
    result = await harness.call(
        "review_pr", {"repo": "tolldog/example", "pr_number": "abc"}
    )
    assert "error" in result


async def test_review_pr_non_positive_pr_number_returns_error():
    provider = _RecordingProvider("ollama", _make_result())
    harness = _make_harness(provider)
    for bad in (0, -1):
        result = await harness.call(
            "review_pr", {"repo": "tolldog/example", "pr_number": bad}
        )
        assert "error" in result


async def test_review_pr_fetch_failure_returns_error():
    provider = _RecordingProvider("ollama", _make_result())
    github = _FakeGithub(
        metadata_error=GithubNotFoundError("get_pr_metadata(tolldog/x#42): 404 Not Found")
    )
    harness = _make_harness(provider, github=github)

    result = await harness.call(
        "review_pr", {"repo": "tolldog/x", "pr_number": 42}
    )

    assert "error" in result
    assert "github fetch failed" in result["error"]
    # provider never called
    assert provider.last_request is None


async def test_review_pr_submit_failure_returns_review_result_plus_error():
    """If posting fails, the caller still sees the review we generated."""
    finding = ReviewFinding(
        severity="nit",
        title="Typo",
        body="x",
        path="f.py",
        line=1,
    )
    provider = _RecordingProvider("ollama", _make_result(findings=[finding]))
    github = _FakeGithub(
        submit_error=GithubClientError("submit_review(x/y#42): 500 Internal Server Error")
    )
    harness = _make_harness(provider, github=github)

    result = await harness.call(
        "review_pr", {"repo": "tolldog/x", "pr_number": 42}
    )

    assert "error" in result
    assert "github post failed" in result["error"]
    # original review still observable
    assert result["summary"] == "ok"
    assert result["findings"][0]["title"] == "Typo"


# ---------------------------------------------------------------------------
# Skill registration
# ---------------------------------------------------------------------------


def test_review_pr_skill_registered():
    provider = _RecordingProvider("ollama", _make_result())
    harness = _make_harness(provider)
    assert "review_pr" in harness.skill_names


def test_review_pr_skill_parameters_match_contract():
    provider = _RecordingProvider("ollama", _make_result())
    harness = _make_harness(provider)
    skill = next(s for s in harness.skills if s.name == "review_pr")
    assert skill.parameters["repo"]["required"] is True
    assert skill.parameters["pr_number"]["required"] is True
    for optional in ("instructions", "backend", "model", "dry_run", "event"):
        assert skill.parameters[optional].get("required", False) is False
