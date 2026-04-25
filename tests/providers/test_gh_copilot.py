"""Tests for ``reviewer.providers.gh_copilot``.

All tests mock :func:`asyncio.create_subprocess_exec` so no real
``copilot`` binary is invoked. The JSONL event shape is the one
``copilot -p --output-format json`` emits as of GitHub Copilot CLI
1.0.36 (verified 2026-04-25).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from khonliang_reviewer import ReviewRequest
from reviewer.providers import gh_copilot
from reviewer.providers.gh_copilot import (
    GhCopilotAuthError,
    GhCopilotProvider,
    GhCopilotProviderConfig,
)


def _final_message_event(content: str, output_tokens: int = 100) -> dict[str, Any]:
    """Build a final-answer assistant.message event the parser expects."""
    return {
        "type": "assistant.message",
        "data": {
            "messageId": "msg-1",
            "content": content,
            "phase": "final_answer",
            "outputTokens": output_tokens,
            "interactionId": "interaction-1",
        },
        "id": "evt-final",
        "timestamp": "2026-04-25T08:26:29.937Z",
    }


def _result_event() -> dict[str, Any]:
    """End-of-stream usage summary event copilot emits."""
    return {
        "type": "result",
        "timestamp": "2026-04-25T08:26:29.954Z",
        "sessionId": "session-1",
        "exitCode": 0,
        "usage": {
            "premiumRequests": 1,
            "totalApiDurationMs": 2407,
            "sessionDurationMs": 4984,
            "codeChanges": {"linesAdded": 0, "linesRemoved": 0, "filesModified": []},
        },
    }


def _success_stdout(payload: dict[str, Any], output_tokens: int = 100) -> bytes:
    """Build a multi-line JSONL stdout that mirrors what copilot emits."""
    events = [
        {"type": "session.mcp_servers_loaded", "data": {"servers": []}},
        {"type": "user.message", "data": {"content": "review this"}},
        {"type": "assistant.turn_start", "data": {"turnId": "0"}},
        _final_message_event(json.dumps(payload), output_tokens=output_tokens),
        {"type": "assistant.turn_end", "data": {"turnId": "0"}},
        _result_event(),
    ]
    return ("\n".join(json.dumps(e) for e in events)).encode()


SUCCESS_PAYLOAD: dict[str, Any] = {
    "summary": "Two findings.",
    "findings": [
        {
            "severity": "concern",
            "title": "Missing tests",
            "body": "No coverage for the empty-input branch.",
            "category": "testing",
            "path": "pkg/mod.py",
            "line": 42,
            "suggestion": None,
        },
        {
            "severity": "nit",
            "title": "Typo",
            "body": "'recieve' -> 'receive'",
            "category": "docs",
            "path": "README.md",
            "line": 10,
            "suggestion": "receive",
        },
    ],
}


class _FakeProc:
    def __init__(
        self,
        *,
        stdout: bytes = b"",
        stderr: bytes = b"",
        returncode: int = 0,
        hang: bool = False,
    ):
        self._stdout = stdout
        self._stderr = stderr
        self.returncode = returncode
        self._hang = hang
        self.killed = False

    async def communicate(self, input: bytes | None = None) -> tuple[bytes, bytes]:
        if self._hang:
            await asyncio.sleep(60)
        return self._stdout, self._stderr

    def kill(self) -> None:
        self.killed = True

    async def wait(self) -> int:
        return self.returncode


def _install_fake_proc(monkeypatch, proc: _FakeProc) -> list[tuple[str, ...]]:
    calls: list[tuple[str, ...]] = []

    async def fake_exec(*cmd: str, **_: Any) -> _FakeProc:
        calls.append(tuple(cmd))
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    return calls


def _install_missing_binary(monkeypatch) -> None:
    async def fake_exec(*_cmd: str, **_: Any) -> _FakeProc:
        raise FileNotFoundError(2, "No such file or directory")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)


def _make_request(**overrides: Any) -> ReviewRequest:
    base: dict[str, Any] = {
        "kind": "pr_diff",
        "content": "diff --git a/x b/x\n@@ -1 +1 @@\n-old\n+new\n",
        "instructions": "Review for correctness.",
        "context": {"repo_profile": "python async bus service"},
        "metadata": {"repo": "tolldog/example", "pr_number": 42},
        "request_id": "req-42",
    }
    base.update(overrides)
    return ReviewRequest(**base)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


async def test_success_payload_produces_posted_review(monkeypatch):
    proc = _FakeProc(stdout=_success_stdout(SUCCESS_PAYLOAD, output_tokens=250))
    calls = _install_fake_proc(monkeypatch, proc)

    result = await GhCopilotProvider().review(_make_request())

    assert result.disposition == "posted"
    assert result.backend == "gh_copilot"
    assert result.model == "copilot"
    assert result.summary == "Two findings."
    assert len(result.findings) == 2
    assert result.findings[0].severity == "concern"
    assert result.findings[0].path == "pkg/mod.py"
    assert result.findings[0].line == 42
    assert result.usage is not None
    assert result.usage.output_tokens == 250
    # Subprocess invoked exactly once with the expected shape.
    assert len(calls) == 1
    argv = calls[0]
    assert argv[0] == "copilot"
    assert "-p" in argv
    assert "--output-format" in argv
    assert "json" in argv
    assert "--allow-all-tools" in argv
    assert "--available-tools=" in argv
    assert "--no-color" in argv
    # The full prompt is in argv[2] (right after `-p`).
    p_idx = argv.index("-p")
    assert "diff --git" in argv[p_idx + 1]


async def test_request_model_threads_to_dash_m(monkeypatch):
    proc = _FakeProc(stdout=_success_stdout(SUCCESS_PAYLOAD))
    calls = _install_fake_proc(monkeypatch, proc)

    request = _make_request(
        metadata={"repo": "tolldog/example", "model": "claude-sonnet-4.5"}
    )
    result = await GhCopilotProvider().review(request)

    assert result.model == "claude-sonnet-4.5"
    argv = calls[0]
    assert "-m" in argv
    m_idx = argv.index("-m")
    assert argv[m_idx + 1] == "claude-sonnet-4.5"


async def test_config_default_model_used_when_request_silent(monkeypatch):
    proc = _FakeProc(stdout=_success_stdout(SUCCESS_PAYLOAD))
    calls = _install_fake_proc(monkeypatch, proc)
    config = GhCopilotProviderConfig(default_model="gpt-5.4")

    result = await GhCopilotProvider(config).review(_make_request())

    assert result.model == "gpt-5.4"
    argv = calls[0]
    m_idx = argv.index("-m")
    assert argv[m_idx + 1] == "gpt-5.4"


async def test_no_model_omits_dash_m(monkeypatch):
    proc = _FakeProc(stdout=_success_stdout(SUCCESS_PAYLOAD))
    calls = _install_fake_proc(monkeypatch, proc)

    await GhCopilotProvider().review(_make_request())

    argv = calls[0]
    assert "-m" not in argv


async def test_reasoning_effort_threaded_to_argv(monkeypatch):
    proc = _FakeProc(stdout=_success_stdout(SUCCESS_PAYLOAD))
    calls = _install_fake_proc(monkeypatch, proc)
    config = GhCopilotProviderConfig(reasoning_effort="high")

    await GhCopilotProvider(config).review(_make_request())

    argv = calls[0]
    assert "--effort" in argv
    e_idx = argv.index("--effort")
    assert argv[e_idx + 1] == "high"


async def test_no_reasoning_effort_omits_flag(monkeypatch):
    proc = _FakeProc(stdout=_success_stdout(SUCCESS_PAYLOAD))
    calls = _install_fake_proc(monkeypatch, proc)

    await GhCopilotProvider().review(_make_request())

    argv = calls[0]
    assert "--effort" not in argv


# ---------------------------------------------------------------------------
# JSONL parser corner cases
# ---------------------------------------------------------------------------


async def test_picks_latest_final_answer_over_earlier(monkeypatch):
    """Two final_answer events → take the latest (turn might iterate)."""
    events = [
        _final_message_event(json.dumps({"summary": "first", "findings": []})),
        _final_message_event(
            json.dumps({"summary": "winner", "findings": []}),
            output_tokens=999,
        ),
        _result_event(),
    ]
    stdout = ("\n".join(json.dumps(e) for e in events)).encode()
    proc = _FakeProc(stdout=stdout)
    _install_fake_proc(monkeypatch, proc)

    result = await GhCopilotProvider().review(_make_request())

    assert result.disposition == "posted"
    assert result.summary == "winner"
    assert result.usage.output_tokens == 999


async def test_falls_back_to_assistant_message_without_phase(monkeypatch):
    """Older CLIs emit assistant.message without a phase field; still parsed."""
    event = {
        "type": "assistant.message",
        "data": {
            "content": json.dumps({"summary": "ok", "findings": []}),
            # no phase field
        },
    }
    stdout = (json.dumps(event) + "\n").encode()
    proc = _FakeProc(stdout=stdout)
    _install_fake_proc(monkeypatch, proc)

    result = await GhCopilotProvider().review(_make_request())

    assert result.disposition == "posted"
    assert result.summary == "ok"


async def test_skips_non_json_lines(monkeypatch):
    """copilot occasionally emits banner/blank lines before JSONL — skipped."""
    payload_event = _final_message_event(json.dumps({"summary": "ok", "findings": []}))
    stdout = (
        b"\n"
        b"GitHub Copilot CLI - banner\n"
        + (json.dumps(payload_event) + "\n").encode()
        + b"\n"
    )
    proc = _FakeProc(stdout=stdout)
    _install_fake_proc(monkeypatch, proc)

    result = await GhCopilotProvider().review(_make_request())

    assert result.disposition == "posted"
    assert result.summary == "ok"


async def test_empty_stdout_errored(monkeypatch):
    proc = _FakeProc(stdout=b"")
    _install_fake_proc(monkeypatch, proc)

    result = await GhCopilotProvider().review(_make_request())

    assert result.disposition == "errored"
    assert result.error_category == "malformed_envelope"
    assert "empty stdout" in result.error


async def test_no_assistant_message_errored(monkeypatch):
    """Stream contains progress events only — no usable response."""
    events = [
        {"type": "session.mcp_servers_loaded", "data": {"servers": []}},
        {"type": "user.message", "data": {"content": "x"}},
        _result_event(),
    ]
    stdout = ("\n".join(json.dumps(e) for e in events)).encode()
    proc = _FakeProc(stdout=stdout)
    _install_fake_proc(monkeypatch, proc)

    result = await GhCopilotProvider().review(_make_request())

    assert result.disposition == "errored"
    assert result.error_category == "malformed_envelope"
    assert "no assistant.message" in result.error


async def test_assistant_message_with_empty_content_errored(monkeypatch):
    event = {
        "type": "assistant.message",
        "data": {"content": "", "phase": "final_answer"},
    }
    proc = _FakeProc(stdout=(json.dumps(event) + "\n").encode())
    _install_fake_proc(monkeypatch, proc)

    result = await GhCopilotProvider().review(_make_request())

    assert result.disposition == "errored"
    assert result.error_category == "malformed_envelope"
    assert "empty content" in result.error


async def test_non_json_content_errored(monkeypatch):
    event = _final_message_event("this is not JSON at all")
    proc = _FakeProc(stdout=(json.dumps(event) + "\n").encode())
    _install_fake_proc(monkeypatch, proc)

    result = await GhCopilotProvider().review(_make_request())

    assert result.disposition == "errored"
    assert result.error_category == "malformed_envelope"
    assert "not JSON" in result.error


async def test_non_object_content_errored(monkeypatch):
    event = _final_message_event(json.dumps(["not", "an", "object"]))
    proc = _FakeProc(stdout=(json.dumps(event) + "\n").encode())
    _install_fake_proc(monkeypatch, proc)

    result = await GhCopilotProvider().review(_make_request())

    assert result.disposition == "errored"
    assert result.error_category == "malformed_envelope"
    assert "not a JSON object" in result.error


# ---------------------------------------------------------------------------
# Defensive payload coercion
# ---------------------------------------------------------------------------


async def test_null_summary_coerced_to_empty(monkeypatch):
    payload = {"summary": None, "findings": []}
    proc = _FakeProc(stdout=_success_stdout(payload))
    _install_fake_proc(monkeypatch, proc)

    result = await GhCopilotProvider().review(_make_request())

    assert result.disposition == "posted"
    assert result.summary == ""


async def test_unknown_severity_coerced_to_comment(monkeypatch):
    payload = {
        "summary": "ok",
        "findings": [
            {"severity": "MEGA", "title": "A", "body": "B"},
            {"severity": None, "title": "C", "body": "D"},
            {"severity": "concern", "title": "E", "body": "F"},
        ],
    }
    proc = _FakeProc(stdout=_success_stdout(payload))
    _install_fake_proc(monkeypatch, proc)

    result = await GhCopilotProvider().review(_make_request())

    severities = [f.severity for f in result.findings]
    assert severities == ["comment", "comment", "concern"]


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


async def test_missing_binary_errored(monkeypatch):
    _install_missing_binary(monkeypatch)

    result = await GhCopilotProvider().review(_make_request())

    assert result.disposition == "errored"
    assert result.error_category == "binary_not_found"
    assert "copilot" in result.error


async def test_argv_too_long_yields_errored_result(monkeypatch):
    """Oversized prompts that overflow argv must surface as errored,
    not crash the skill call. ``OSError`` (errno 7 / E2BIG) is the
    most likely concrete cause; this test simulates it generically."""

    async def fake_exec(*_cmd: str, **_kwargs: object) -> object:
        # OSError with errno=7 mirrors how Linux reports E2BIG when
        # execve's argv exceeds ARG_MAX. The provider's broad
        # OSError catch covers this branch without depending on the
        # specific errno — but we set it so the error message
        # surfaces realistic context.
        err = OSError(7, "Argument list too long")
        raise err

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    # Use a large prompt so the byte-count in the error message is
    # representative of the failure mode operators would see.
    big = _make_request(content="x" * 200_000)
    result = await GhCopilotProvider().review(big)

    assert result.disposition == "errored"
    assert result.error_category == "backend_error"
    assert "errno=7" in result.error
    # Prompt-size hint must be in the operator-facing message so
    # the trigger condition is visible without re-running.
    assert "bytes" in result.error
    assert "ARG_MAX" in result.error


async def test_subprocess_timeout_errored(monkeypatch):
    proc = _FakeProc(
        stdout=_success_stdout(SUCCESS_PAYLOAD),
        hang=True,
    )
    _install_fake_proc(monkeypatch, proc)

    config = GhCopilotProviderConfig(timeout_seconds=0.05)
    result = await GhCopilotProvider(config).review(_make_request())

    assert result.disposition == "errored"
    assert result.error_category == "subprocess_timeout"
    assert proc.killed


async def test_non_zero_exit_errored(monkeypatch):
    proc = _FakeProc(stdout=b"", stderr=b"kaboom", returncode=2)
    _install_fake_proc(monkeypatch, proc)

    result = await GhCopilotProvider().review(_make_request())

    assert result.disposition == "errored"
    assert result.error_category == "nonzero_exit"
    assert "kaboom" in result.error


async def test_non_zero_exit_with_auth_hint_upgrades_category(monkeypatch):
    proc = _FakeProc(
        stdout=b"",
        stderr=b"Error: 401 unauthorized - please log in",
        returncode=1,
    )
    _install_fake_proc(monkeypatch, proc)

    result = await GhCopilotProvider().review(_make_request())

    assert result.disposition == "errored"
    assert result.error_category == "auth_not_provisioned"


# ---------------------------------------------------------------------------
# Healthcheck
# ---------------------------------------------------------------------------


async def test_healthcheck_with_binary_and_token_succeeds(monkeypatch):
    monkeypatch.setattr(
        gh_copilot.shutil, "which",
        lambda binary: "/usr/local/bin/copilot" if binary == "copilot" else None,
    )
    # Use a v2 fine-grained PAT shape (``github_pat_*``); copilot
    # does NOT accept classic ``ghp_`` PATs, so the test fixture
    # should reflect what an operator would actually set.
    monkeypatch.setenv("GH_TOKEN", "github_pat_fake")

    # Should not raise.
    await GhCopilotProvider().healthcheck()


async def test_healthcheck_with_binary_and_oauth_dir_succeeds(monkeypatch, tmp_path):
    monkeypatch.setattr(
        gh_copilot.shutil, "which",
        lambda binary: "/usr/local/bin/copilot" if binary == "copilot" else None,
    )
    fake_home = tmp_path / "home"
    (fake_home / ".copilot").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(fake_home))
    for env in ("COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"):
        monkeypatch.delenv(env, raising=False)

    await GhCopilotProvider().healthcheck()


async def test_healthcheck_no_binary_raises(monkeypatch):
    monkeypatch.setattr(gh_copilot.shutil, "which", lambda _b: None)

    with pytest.raises(FileNotFoundError):
        await GhCopilotProvider().healthcheck()


async def test_healthcheck_no_auth_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(
        gh_copilot.shutil, "which",
        lambda binary: "/usr/local/bin/copilot" if binary == "copilot" else None,
    )
    fresh = tmp_path / "no-copilot"
    fresh.mkdir()
    monkeypatch.setenv("HOME", str(fresh))
    for env in ("COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"):
        monkeypatch.delenv(env, raising=False)

    with pytest.raises(GhCopilotAuthError) as excinfo:
        await GhCopilotProvider().healthcheck()
    assert "copilot login" in str(excinfo.value)
    assert "GH_TOKEN" in str(excinfo.value)


def test_provider_name_is_gh_copilot():
    assert GhCopilotProvider.name == "gh_copilot"
    assert GhCopilotProvider().name == "gh_copilot"


def test_auth_error_subclasses_runtime_error():
    """Callers catching RuntimeError pick up auth failures broadly."""
    assert issubclass(GhCopilotAuthError, RuntimeError)
