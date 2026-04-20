"""Tests for ``reviewer.providers.claude_cli``.

All tests mock :func:`asyncio.create_subprocess_exec` so no real ``claude``
binary is invoked. The envelope shape is the one emitted by the current
Claude Code ``-p --output-format=json`` mode.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from khonliang_reviewer import ReviewRequest
from reviewer.providers import claude_cli
from reviewer.providers.claude_cli import (
    ClaudeCliProvider,
    ClaudeCliProviderConfig,
)


SUCCESS_ENVELOPE: dict[str, Any] = {
    "type": "result",
    "subtype": "success",
    "is_error": False,
    "api_error_status": None,
    "duration_ms": 1591,
    "result": json.dumps(
        {
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
    ),
    "total_cost_usd": 0.12345,
    "usage": {
        "input_tokens": 5,
        "output_tokens": 250,
        "cache_read_input_tokens": 16000,
        "cache_creation_input_tokens": 17000,
    },
    "modelUsage": {
        "claude-haiku-4-5": {"costUSD": 0.0003},
        "claude-opus-4-7[1m]": {"costUSD": 0.12315},
    },
}


class _FakeProc:
    """Minimal stand-in for :class:`asyncio.subprocess.Process`."""

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

    async def communicate(self) -> tuple[bytes, bytes]:
        if self._hang:
            await asyncio.sleep(60)
        return self._stdout, self._stderr

    def kill(self) -> None:
        self.killed = True

    async def wait(self) -> int:
        return self.returncode


def _install_fake_proc(monkeypatch, proc: _FakeProc) -> list[tuple[str, ...]]:
    """Monkeypatch ``create_subprocess_exec`` to return ``proc``.

    Returns the call log (list of argv tuples) so tests can assert on the
    exact command that would have been spawned.
    """
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


async def test_success_envelope_produces_posted_review(monkeypatch):
    proc = _FakeProc(stdout=json.dumps(SUCCESS_ENVELOPE).encode())
    calls = _install_fake_proc(monkeypatch, proc)
    provider = ClaudeCliProvider()

    result = await provider.review(_make_request())

    assert result.disposition == "posted"
    assert result.backend == "claude_cli"
    # highest-cost model is picked as the primary
    assert result.model == "claude-opus-4-7[1m]"
    assert result.summary == "Two findings."
    assert len(result.findings) == 2
    first = result.findings[0]
    assert first.severity == "concern"
    assert first.path == "pkg/mod.py"
    assert first.line == 42
    assert result.usage is not None
    assert result.usage.input_tokens == 5
    assert result.usage.output_tokens == 250
    assert result.usage.cache_read_tokens == 16000
    assert result.usage.cache_creation_tokens == 17000
    assert result.usage.duration_ms == 1591
    assert result.usage.estimated_api_cost_usd == pytest.approx(0.12345)
    assert result.usage.repo == "tolldog/example"
    assert result.usage.pr_number == 42
    assert result.usage.disposition == "posted"
    # exactly one subprocess call with the expected shape
    assert len(calls) == 1
    argv = calls[0]
    assert argv[0] == "claude"
    assert "-p" in argv
    assert "--output-format=json" in argv
    assert "--json-schema" in argv


async def test_prompt_carries_instructions_and_context(monkeypatch):
    proc = _FakeProc(stdout=json.dumps(SUCCESS_ENVELOPE).encode())
    calls = _install_fake_proc(monkeypatch, proc)

    await ClaudeCliProvider().review(_make_request())

    prompt = calls[0][-1]  # last argv element is the positional prompt
    assert "Review for correctness." in prompt
    assert "python async bus service" in prompt
    assert "diff --git" in prompt
    # Schema JSON is passed via --json-schema, not embedded in the prompt
    assert '"severity"' not in prompt


async def test_error_envelope_errored_disposition(monkeypatch):
    error_envelope = {
        "type": "result",
        "is_error": True,
        "api_error_status": "rate_limited",
        "result": "",
        "duration_ms": 12,
        "usage": {},
    }
    proc = _FakeProc(stdout=json.dumps(error_envelope).encode())
    _install_fake_proc(monkeypatch, proc)

    result = await ClaudeCliProvider().review(_make_request())

    assert result.disposition == "errored"
    assert "rate_limited" in result.error
    assert result.usage is not None
    assert result.usage.disposition == "errored"


async def test_non_zero_exit_code_errored(monkeypatch):
    proc = _FakeProc(stdout=b"", stderr=b"kaboom", returncode=2)
    _install_fake_proc(monkeypatch, proc)

    result = await ClaudeCliProvider().review(_make_request())

    assert result.disposition == "errored"
    assert "exited with 2" in result.error
    assert "kaboom" in result.error


async def test_non_json_stdout_errored(monkeypatch):
    proc = _FakeProc(stdout=b"not json at all")
    _install_fake_proc(monkeypatch, proc)

    result = await ClaudeCliProvider().review(_make_request())

    assert result.disposition == "errored"
    assert "non-JSON output" in result.error


async def test_result_field_non_json_errored(monkeypatch):
    """Envelope is valid JSON but the ``result`` string isn't."""
    envelope = dict(SUCCESS_ENVELOPE)
    envelope["result"] = "not json"
    proc = _FakeProc(stdout=json.dumps(envelope).encode())
    _install_fake_proc(monkeypatch, proc)

    result = await ClaudeCliProvider().review(_make_request())

    assert result.disposition == "errored"
    assert "result was not JSON" in result.error


async def test_missing_binary_errored(monkeypatch):
    _install_missing_binary(monkeypatch)
    provider = ClaudeCliProvider(ClaudeCliProviderConfig(binary="claude-does-not-exist"))

    result = await provider.review(_make_request())

    assert result.disposition == "errored"
    assert "claude-does-not-exist" in result.error
    assert "not found" in result.error


async def test_timeout_errored_and_kills_process(monkeypatch):
    proc = _FakeProc(stdout=b"", hang=True)
    _install_fake_proc(monkeypatch, proc)
    provider = ClaudeCliProvider(ClaudeCliProviderConfig(timeout_seconds=0.01))

    result = await provider.review(_make_request())

    assert result.disposition == "errored"
    assert "timed out" in result.error
    assert proc.killed is True


async def test_primary_model_is_highest_cost(monkeypatch):
    envelope = dict(SUCCESS_ENVELOPE)
    envelope["modelUsage"] = {
        "small-model": {"costUSD": 0.01},
        "big-model": {"costUSD": 1.23},
        "medium-model": {"costUSD": 0.5},
    }
    proc = _FakeProc(stdout=json.dumps(envelope).encode())
    _install_fake_proc(monkeypatch, proc)

    result = await ClaudeCliProvider().review(_make_request())

    assert result.model == "big-model"


async def test_missing_model_usage_falls_back_to_claude(monkeypatch):
    envelope = dict(SUCCESS_ENVELOPE)
    envelope.pop("modelUsage", None)
    proc = _FakeProc(stdout=json.dumps(envelope).encode())
    _install_fake_proc(monkeypatch, proc)

    result = await ClaudeCliProvider().review(_make_request())

    assert result.model == "claude"


async def test_total_cost_usd_maps_to_estimated_api_cost(monkeypatch):
    """The envelope's top-level cost IS the API-equivalent cost."""
    envelope = dict(SUCCESS_ENVELOPE)
    envelope["total_cost_usd"] = 7.89
    proc = _FakeProc(stdout=json.dumps(envelope).encode())
    _install_fake_proc(monkeypatch, proc)

    result = await ClaudeCliProvider().review(_make_request())

    assert result.usage is not None
    assert result.usage.estimated_api_cost_usd == pytest.approx(7.89)


async def test_append_system_prompt_flag_threaded_through(monkeypatch):
    proc = _FakeProc(stdout=json.dumps(SUCCESS_ENVELOPE).encode())
    calls = _install_fake_proc(monkeypatch, proc)
    provider = ClaudeCliProvider(
        ClaudeCliProviderConfig(append_system_prompt="You are a reviewer.")
    )

    await provider.review(_make_request())

    argv = calls[0]
    assert "--append-system-prompt" in argv
    idx = argv.index("--append-system-prompt")
    assert argv[idx + 1] == "You are a reviewer."


async def test_provider_name_is_claude_cli():
    assert ClaudeCliProvider.name == "claude_cli"
    assert ClaudeCliProvider().name == "claude_cli"


async def test_review_response_schema_shape():
    schema = claude_cli.REVIEW_RESPONSE_SCHEMA
    assert schema["type"] == "object"
    assert "summary" in schema["required"]
    severity = schema["properties"]["findings"]["items"]["properties"]["severity"]
    assert set(severity["enum"]) == {"nit", "comment", "concern"}
