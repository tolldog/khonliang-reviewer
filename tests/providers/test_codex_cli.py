"""Tests for ``reviewer.providers.codex_cli``.

All tests mock :func:`asyncio.create_subprocess_exec` so no real
``codex`` binary is invoked. The output shape is what
``codex exec --output-schema`` writes to stdout: a single JSON object
matching the REVIEW_RESPONSE_SCHEMA.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

import pytest

from khonliang_reviewer import ReviewRequest
from reviewer.providers import codex_cli
from reviewer.providers.codex_cli import (
    CodexCliAuthError,
    CodexCliProvider,
    CodexCliProviderConfig,
)


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
        self.stdin_received: bytes | None = None

    async def communicate(self, input: bytes | None = None) -> tuple[bytes, bytes]:
        self.stdin_received = input
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


async def test_success_payload_produces_posted_review(monkeypatch):
    proc = _FakeProc(stdout=json.dumps(SUCCESS_PAYLOAD).encode())
    calls = _install_fake_proc(monkeypatch, proc)
    provider = CodexCliProvider()

    result = await provider.review(_make_request())

    assert result.disposition == "posted"
    assert result.backend == "codex_cli"
    # No request-supplied model and config has empty default -> falls
    # back to "codex" placeholder for ReviewResult.model.
    assert result.model == "codex"
    assert result.summary == "Two findings."
    assert len(result.findings) == 2
    first = result.findings[0]
    assert first.severity == "concern"
    assert first.path == "pkg/mod.py"
    assert first.line == 42
    assert result.usage is not None
    assert result.usage.duration_ms >= 0
    assert result.usage.disposition == "posted"
    assert result.usage.repo == "tolldog/example"
    assert result.usage.pr_number == 42
    # Token fields are 0 — codex exec without --json doesn't emit counts.
    assert result.usage.input_tokens == 0
    assert result.usage.output_tokens == 0
    # exactly one subprocess call with the expected shape
    assert len(calls) == 1
    argv = calls[0]
    assert argv[0] == "codex"
    assert argv[1] == "exec"
    assert "--ephemeral" in argv
    assert "--skip-git-repo-check" in argv
    assert "--ignore-user-config" in argv
    assert "--ignore-rules" in argv
    assert "--output-schema" in argv
    # Final argv is "-" (read prompt from stdin)
    assert argv[-1] == "-"
    # No -m flag because neither request nor config supplied a model
    assert "-m" not in argv
    # prompt is NOT in argv — it's piped via stdin
    assert not any("diff --git" in part for part in argv)
    assert proc.stdin_received is not None
    assert b"diff --git" in proc.stdin_received


async def test_request_model_overrides_default(monkeypatch):
    proc = _FakeProc(stdout=json.dumps(SUCCESS_PAYLOAD).encode())
    calls = _install_fake_proc(monkeypatch, proc)
    provider = CodexCliProvider()

    request = _make_request(metadata={"repo": "tolldog/example", "model": "gpt-5"})
    result = await provider.review(request)

    assert result.model == "gpt-5"
    argv = calls[0]
    assert "-m" in argv
    m_idx = argv.index("-m")
    assert argv[m_idx + 1] == "gpt-5"


async def test_config_default_model_used_when_request_silent(monkeypatch):
    proc = _FakeProc(stdout=json.dumps(SUCCESS_PAYLOAD).encode())
    calls = _install_fake_proc(monkeypatch, proc)
    provider = CodexCliProvider(CodexCliProviderConfig(default_model="o3"))

    result = await provider.review(_make_request())

    assert result.model == "o3"
    argv = calls[0]
    assert "-m" in argv
    m_idx = argv.index("-m")
    assert argv[m_idx + 1] == "o3"


async def test_prompt_carries_instructions_and_context(monkeypatch):
    proc = _FakeProc(stdout=json.dumps(SUCCESS_PAYLOAD).encode())
    _install_fake_proc(monkeypatch, proc)

    await CodexCliProvider().review(_make_request())

    assert proc.stdin_received is not None
    prompt = proc.stdin_received.decode()
    assert "Review for correctness." in prompt
    assert "python async bus service" in prompt
    assert "diff --git" in prompt
    # Schema enforced via --output-schema flag, not embedded in the prompt
    assert '"severity"' not in prompt


async def test_large_prompt_survives_via_stdin(monkeypatch):
    """Prompts larger than typical ARG_MAX must not touch argv."""
    proc = _FakeProc(stdout=json.dumps(SUCCESS_PAYLOAD).encode())
    calls = _install_fake_proc(monkeypatch, proc)

    big_diff = "diff --git a/f b/f\n" + ("+line\n" * 30_000)
    request = _make_request(content=big_diff)

    await CodexCliProvider().review(request)

    argv = calls[0]
    assert all(len(part.encode()) < 50_000 for part in argv)
    assert proc.stdin_received is not None
    assert len(proc.stdin_received) > 100_000


async def test_empty_stdout_errored(monkeypatch):
    proc = _FakeProc(stdout=b"")
    _install_fake_proc(monkeypatch, proc)

    result = await CodexCliProvider().review(_make_request())

    assert result.disposition == "errored"
    assert result.error_category == "malformed_envelope"
    assert "empty stdout" in result.error


async def test_non_json_stdout_errored(monkeypatch):
    proc = _FakeProc(stdout=b"not json at all")
    _install_fake_proc(monkeypatch, proc)

    result = await CodexCliProvider().review(_make_request())

    assert result.disposition == "errored"
    assert result.error_category == "malformed_envelope"
    assert "non-JSON output" in result.error


async def test_non_object_payload_errored(monkeypatch):
    """Schema enforcement should always yield an object, but be defensive."""
    proc = _FakeProc(stdout=b'["not", "an", "object"]')
    _install_fake_proc(monkeypatch, proc)

    result = await CodexCliProvider().review(_make_request())

    assert result.disposition == "errored"
    assert result.error_category == "malformed_envelope"
    assert "not an object" in result.error


async def test_non_zero_exit_code_errored(monkeypatch):
    proc = _FakeProc(stdout=b"", stderr=b"kaboom", returncode=2)
    _install_fake_proc(monkeypatch, proc)

    result = await CodexCliProvider().review(_make_request())

    assert result.disposition == "errored"
    assert result.error_category == "nonzero_exit"
    assert "exited with 2" in result.error
    assert "kaboom" in result.error


async def test_non_zero_exit_with_auth_hint_upgrades_category(monkeypatch):
    """Mid-session auth revocation should be categorized specifically."""
    proc = _FakeProc(
        stdout=b"",
        stderr=b"Error: not authenticated; please log in",
        returncode=1,
    )
    _install_fake_proc(monkeypatch, proc)

    result = await CodexCliProvider().review(_make_request())

    assert result.disposition == "errored"
    assert result.error_category == "auth_not_provisioned"


async def test_missing_binary_errored(monkeypatch):
    _install_missing_binary(monkeypatch)

    result = await CodexCliProvider().review(_make_request())

    assert result.disposition == "errored"
    assert result.error_category == "binary_not_found"
    assert "codex" in result.error


async def test_subprocess_timeout_errored(monkeypatch):
    proc = _FakeProc(stdout=json.dumps(SUCCESS_PAYLOAD).encode(), hang=True)
    _install_fake_proc(monkeypatch, proc)

    config = CodexCliProviderConfig(timeout_seconds=0.05)
    result = await CodexCliProvider(config).review(_make_request())

    assert result.disposition == "errored"
    assert result.error_category == "subprocess_timeout"
    assert "timed out" in result.error
    assert proc.killed


async def test_findings_with_missing_optional_fields(monkeypatch):
    """A finding with only required keys (severity/title/body) should parse."""
    payload = {
        "summary": "One finding.",
        "findings": [
            {
                "severity": "comment",
                "title": "Note",
                "body": "Heads up.",
            }
        ],
    }
    proc = _FakeProc(stdout=json.dumps(payload).encode())
    _install_fake_proc(monkeypatch, proc)

    result = await CodexCliProvider().review(_make_request())

    assert result.disposition == "posted"
    assert len(result.findings) == 1
    finding = result.findings[0]
    assert finding.severity == "comment"
    assert finding.title == "Note"
    assert finding.body == "Heads up."
    assert finding.path is None
    assert finding.line is None
    assert finding.suggestion is None
    assert finding.category == ""


async def test_payload_without_findings_yields_empty_list(monkeypatch):
    """``findings`` is optional in the schema — absence becomes ``[]``."""
    payload = {"summary": "Looks fine."}
    proc = _FakeProc(stdout=json.dumps(payload).encode())
    _install_fake_proc(monkeypatch, proc)

    result = await CodexCliProvider().review(_make_request())

    assert result.disposition == "posted"
    assert result.summary == "Looks fine."
    assert result.findings == []


async def test_null_summary_coerced_to_empty_string(monkeypatch):
    """An off-spec payload with ``summary: null`` must not produce ``"None"``."""
    proc = _FakeProc(stdout=b'{"summary": null, "findings": []}')
    _install_fake_proc(monkeypatch, proc)

    result = await CodexCliProvider().review(_make_request())

    assert result.disposition == "posted"
    assert result.summary == ""
    assert result.findings == []


async def test_unknown_severity_coerced_to_comment(monkeypatch):
    """Severities outside the contract enum default to ``comment`` — preserves
    severity_floor filtering and avoids leaking off-spec values into ReviewFinding.
    """
    payload = {
        "summary": "ok",
        "findings": [
            {"severity": "MEGA", "title": "A", "body": "B"},
            {"severity": None, "title": "C", "body": "D"},
            {"severity": 7, "title": "E", "body": "F"},
            {"severity": "concern", "title": "G", "body": "H"},
        ],
    }
    proc = _FakeProc(stdout=json.dumps(payload).encode())
    _install_fake_proc(monkeypatch, proc)

    result = await CodexCliProvider().review(_make_request())

    assert result.disposition == "posted"
    severities = [f.severity for f in result.findings]
    assert severities == ["comment", "comment", "comment", "concern"]


async def test_null_finding_string_fields_coerced(monkeypatch):
    """Null title / body / category / path / suggestion must not stringify to ``"None"``."""
    payload = {
        "summary": "ok",
        "findings": [
            {
                "severity": "nit",
                "title": None,
                "body": None,
                "category": None,
                "path": None,
                "line": None,
                "suggestion": None,
            },
        ],
    }
    proc = _FakeProc(stdout=json.dumps(payload).encode())
    _install_fake_proc(monkeypatch, proc)

    result = await CodexCliProvider().review(_make_request())

    finding = result.findings[0]
    assert finding.title == ""
    assert finding.body == ""
    assert finding.category == ""
    assert finding.path is None
    assert finding.line is None
    assert finding.suggestion is None


async def test_healthcheck_logged_in_succeeds(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    proc = _FakeProc(stdout=b"Logged in using ChatGPT\n")
    _install_fake_proc(monkeypatch, proc)

    # Should not raise
    await CodexCliProvider().healthcheck()


async def test_healthcheck_logged_out_raises_auth_error(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    proc = _FakeProc(stdout=b"Not logged in\n")
    _install_fake_proc(monkeypatch, proc)

    with pytest.raises(CodexCliAuthError):
        await CodexCliProvider().healthcheck()


async def test_healthcheck_api_key_env_skips_login_probe(monkeypatch):
    """OPENAI_API_KEY presence accepts the env-var auth path without invoking codex login."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    calls: list[tuple[str, ...]] = []

    async def fake_exec(*cmd: str, **_: object) -> _FakeProc:
        calls.append(tuple(cmd))
        return _FakeProc(stdout=b"")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    # Should not raise even though no login probe runs
    await CodexCliProvider().healthcheck()
    assert calls == [], "healthcheck should short-circuit on OPENAI_API_KEY"


async def test_healthcheck_nonzero_exit_raises_runtime_error(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    proc = _FakeProc(stdout=b"", stderr=b"oops", returncode=1)
    _install_fake_proc(monkeypatch, proc)

    with pytest.raises(RuntimeError) as excinfo:
        await CodexCliProvider().healthcheck()
    assert "exited with 1" in str(excinfo.value)


async def test_healthcheck_missing_binary_raises_filenotfound(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    _install_missing_binary(monkeypatch)

    with pytest.raises(FileNotFoundError):
        await CodexCliProvider().healthcheck()


def test_schema_file_lazy_init_writes_on_first_use():
    """Schema path is None at construction; first access materializes the file."""
    provider = CodexCliProvider()
    # Eager-init regression guard: __init__ must NOT touch the disk.
    assert provider._schema_path is None
    path = provider._get_schema_path()
    assert os.path.isfile(path)
    # Calls after the first reuse the cached path (no rewrite).
    assert provider._get_schema_path() == path
    with open(path) as f:
        loaded = json.load(f)
    assert loaded == codex_cli.REVIEW_RESPONSE_SCHEMA
    finding_props = loaded["properties"]["findings"]["items"]["properties"]
    assert finding_props["severity"]["enum"] == ["nit", "comment", "concern"]


def test_schema_file_uses_unique_tempfile_path():
    """Two provider instances must materialize to distinct paths (mkstemp guarantees uniqueness)."""
    a = CodexCliProvider()
    b = CodexCliProvider()
    assert a._get_schema_path() != b._get_schema_path()


async def test_schema_materialization_oserror_yields_errored_result(monkeypatch):
    """A tempfile/disk failure during lazy schema init should not crash review()."""
    provider = CodexCliProvider()

    def fail_materialize() -> str:
        raise OSError("No space left on device")

    monkeypatch.setattr(provider, "_get_schema_path", fail_materialize)

    # The fake subprocess never gets called because the schema-write
    # error short-circuits review() before subprocess construction.
    calls: list[tuple[str, ...]] = []

    async def fake_exec(*cmd: str, **_: Any) -> _FakeProc:
        calls.append(tuple(cmd))
        return _FakeProc(stdout=json.dumps(SUCCESS_PAYLOAD).encode())

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    result = await provider.review(_make_request())

    assert result.disposition == "errored"
    assert result.error_category == "backend_error"
    assert "output-schema file" in result.error
    assert "No space left on device" in result.error
    # Subprocess must NOT have been spawned — short-circuit before argv.
    assert calls == []


# Integration: exercises the real codex binary if available + authenticated.
@pytest.mark.skipif(
    not os.path.exists(os.path.expanduser("~/.codex/auth.json")),
    reason="codex auth.json missing — skipping live integration test",
)
async def test_live_codex_review_smoke():
    """Smoke test: real codex on a tiny synthetic diff returns a result."""
    import shutil

    if shutil.which("codex") is None:
        pytest.skip("codex binary not on PATH")

    request = ReviewRequest(
        kind="pr_diff",
        content="diff --git a/x.py b/x.py\n@@ -1,1 +1,1 @@\n-a = 1\n+a = 2\n",
        instructions="Review this trivial change. Return a short JSON object.",
        request_id="codex-smoke",
    )
    # Generous timeout — codex exec end-to-end can take 30s+ on the
    # network-bound subscription path.
    config = CodexCliProviderConfig(timeout_seconds=180.0)
    result = await CodexCliProvider(config).review(request)

    # Either posted (model returned schema-conforming JSON) or errored
    # (network/auth/subscription issue) — both are valid outcomes for a
    # smoke test. The point is the pipe ran end-to-end without crashing.
    assert result.disposition in ("posted", "errored")
    assert result.backend == "codex_cli"
    if result.disposition == "posted":
        assert isinstance(result.summary, str)
        # findings may be empty for trivial diffs
        for f in result.findings:
            assert f.severity in ("nit", "comment", "concern")
