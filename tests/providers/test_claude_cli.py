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
    ClaudeCliAuthError,
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
    # Sub-Claude has no need for tools; --permission-mode dontAsk denies
    # anything outside permissions.allow + the read-only command set.
    assert "--permission-mode" in argv
    pm_idx = argv.index("--permission-mode")
    assert argv[pm_idx + 1] == "dontAsk"
    # prompt is NOT in argv — it's piped via stdin to avoid ARG_MAX and
    # the `ps`-listing leak for diff content
    assert not any("diff --git" in part for part in argv)
    assert proc.stdin_received is not None
    assert b"diff --git" in proc.stdin_received


async def test_prompt_carries_instructions_and_context(monkeypatch):
    proc = _FakeProc(stdout=json.dumps(SUCCESS_ENVELOPE).encode())
    _install_fake_proc(monkeypatch, proc)

    await ClaudeCliProvider().review(_make_request())

    # Prompt is piped via stdin, not argv
    assert proc.stdin_received is not None
    prompt = proc.stdin_received.decode()
    assert "Review for correctness." in prompt
    assert "python async bus service" in prompt
    assert "diff --git" in prompt
    # Schema JSON is passed via --json-schema flag, not embedded in the prompt
    assert '"severity"' not in prompt


async def test_large_prompt_survives_via_stdin(monkeypatch):
    """Prompts larger than typical ARG_MAX must not touch argv."""
    proc = _FakeProc(stdout=json.dumps(SUCCESS_ENVELOPE).encode())
    calls = _install_fake_proc(monkeypatch, proc)

    # 150KB diff — comfortably above Linux's default 128KB ARG_MAX per arg
    big_diff = "diff --git a/f b/f\n" + ("+line\n" * 30_000)
    request = _make_request(content=big_diff)

    await ClaudeCliProvider().review(request)

    argv = calls[0]
    # None of the argv entries contain the large diff content
    assert all(len(part.encode()) < 50_000 for part in argv)
    assert proc.stdin_received is not None
    assert len(proc.stdin_received) > 100_000


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
    assert result.error_category == "backend_error"
    assert "rate_limited" in result.error
    assert result.usage is not None
    assert result.usage.disposition == "errored"
    assert result.usage.error_category == "backend_error"


async def test_non_zero_exit_code_errored(monkeypatch):
    proc = _FakeProc(stdout=b"", stderr=b"kaboom", returncode=2)
    _install_fake_proc(monkeypatch, proc)

    result = await ClaudeCliProvider().review(_make_request())

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

    result = await ClaudeCliProvider().review(_make_request())

    assert result.disposition == "errored"
    assert result.error_category == "auth_not_provisioned"


async def test_non_zero_exit_with_unknown_option_rewrites_message(monkeypatch):
    """Older claude CLIs that don't recognize --permission-mode get a clear diagnostic.

    Category stays ``nonzero_exit`` because the binary is present and
    ran — the ErrorCategory enum has no ``binary_incompatible`` slot
    today. The operator-facing message is what carries the version
    requirement and the right config knob to update.
    """
    proc = _FakeProc(
        stdout=b"",
        stderr=b"error: unknown option '--permission-mode'",
        returncode=2,
    )
    _install_fake_proc(monkeypatch, proc)

    result = await ClaudeCliProvider().review(_make_request())

    assert result.disposition == "errored"
    # Same category as a generic non-zero exit; analytics see the
    # technical truth (CLI exited non-zero), operators get the
    # diagnostic message instead.
    assert result.error_category == "nonzero_exit"
    assert "rejected an argument" in result.error
    assert ">= 2.1.119" in result.error
    # Operator pointer references the actual config knob, not a
    # nonexistent env var.
    assert "providers.claude_cli.binary" in result.error
    # Original stderr is preserved so operators can still see the
    # underlying CLI message for context.
    assert "unknown option" in result.error


async def test_non_json_stdout_errored(monkeypatch):
    proc = _FakeProc(stdout=b"not json at all")
    _install_fake_proc(monkeypatch, proc)

    result = await ClaudeCliProvider().review(_make_request())

    assert result.disposition == "errored"
    assert result.error_category == "malformed_envelope"
    assert "non-JSON output" in result.error


async def test_result_field_non_json_errored(monkeypatch):
    """Envelope is valid JSON but the ``result`` string isn't."""
    envelope = dict(SUCCESS_ENVELOPE)
    envelope["result"] = "not json"
    proc = _FakeProc(stdout=json.dumps(envelope).encode())
    _install_fake_proc(monkeypatch, proc)

    result = await ClaudeCliProvider().review(_make_request())

    assert result.disposition == "errored"
    assert result.error_category == "malformed_envelope"
    assert "result was not JSON" in result.error


async def test_structured_output_is_preferred_over_result(monkeypatch):
    """With --json-schema the CLI fills structured_output; result is empty."""
    envelope = dict(SUCCESS_ENVELOPE)
    envelope["result"] = ""
    envelope["structured_output"] = {
        "summary": "Structured summary.",
        "findings": [
            {
                "severity": "nit",
                "title": "From schema",
                "body": "x",
                "path": "a.py",
                "line": 5,
            }
        ],
    }
    proc = _FakeProc(stdout=json.dumps(envelope).encode())
    _install_fake_proc(monkeypatch, proc)

    result = await ClaudeCliProvider().review(_make_request())

    assert result.disposition == "posted"
    assert result.summary == "Structured summary."
    assert len(result.findings) == 1
    assert result.findings[0].title == "From schema"


async def test_structured_output_takes_precedence_when_both_populated(monkeypatch):
    """Belt-and-suspenders: if both fields are present, structured_output wins."""
    envelope = dict(SUCCESS_ENVELOPE)
    envelope["structured_output"] = {"summary": "from schema", "findings": []}
    proc = _FakeProc(stdout=json.dumps(envelope).encode())
    _install_fake_proc(monkeypatch, proc)

    result = await ClaudeCliProvider().review(_make_request())

    assert result.disposition == "posted"
    assert result.summary == "from schema"


async def test_non_dict_structured_output_falls_back_to_result(monkeypatch):
    """Malformed structured_output falls back to parsing the result string."""
    envelope = dict(SUCCESS_ENVELOPE)
    envelope["structured_output"] = ["not", "a", "dict"]
    proc = _FakeProc(stdout=json.dumps(envelope).encode())
    _install_fake_proc(monkeypatch, proc)

    result = await ClaudeCliProvider().review(_make_request())

    # SUCCESS_ENVELOPE's result is valid JSON; fallback succeeds.
    assert result.disposition == "posted"
    assert result.summary == "Two findings."


async def test_missing_binary_errored(monkeypatch):
    _install_missing_binary(monkeypatch)
    provider = ClaudeCliProvider(ClaudeCliProviderConfig(binary="claude-does-not-exist"))

    result = await provider.review(_make_request())

    assert result.disposition == "errored"
    assert result.error_category == "binary_not_found"
    assert "claude-does-not-exist" in result.error
    assert "not found" in result.error


async def test_timeout_errored_and_kills_process(monkeypatch):
    proc = _FakeProc(stdout=b"", hang=True)
    _install_fake_proc(monkeypatch, proc)
    provider = ClaudeCliProvider(ClaudeCliProviderConfig(timeout_seconds=0.01))

    result = await provider.review(_make_request())

    assert result.disposition == "errored"
    assert result.error_category == "subprocess_timeout"
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


async def test_model_from_metadata_threaded_as_cli_flag(monkeypatch):
    """Caller-specified model (via request.metadata) reaches claude -p --model."""
    proc = _FakeProc(stdout=json.dumps(SUCCESS_ENVELOPE).encode())
    calls = _install_fake_proc(monkeypatch, proc)

    request = _make_request(
        metadata={"repo": "tolldog/ex", "pr_number": 7, "model": "sonnet"}
    )
    await ClaudeCliProvider().review(request)

    argv = calls[0]
    assert "--model" in argv
    idx = argv.index("--model")
    assert argv[idx + 1] == "sonnet"


async def test_no_model_in_metadata_omits_model_flag(monkeypatch):
    """Without a caller-specified model, no --model flag is passed."""
    proc = _FakeProc(stdout=json.dumps(SUCCESS_ENVELOPE).encode())
    calls = _install_fake_proc(monkeypatch, proc)

    request = _make_request(metadata={"repo": "tolldog/ex", "pr_number": 7})
    await ClaudeCliProvider().review(request)

    argv = calls[0]
    assert "--model" not in argv


async def test_provider_name_is_claude_cli():
    assert ClaudeCliProvider.name == "claude_cli"
    assert ClaudeCliProvider().name == "claude_cli"


async def test_review_response_schema_shape():
    schema = claude_cli.REVIEW_RESPONSE_SCHEMA
    assert schema["type"] == "object"
    assert "summary" in schema["required"]
    severity = schema["properties"]["findings"]["items"]["properties"]["severity"]
    assert set(severity["enum"]) == {"nit", "comment", "concern"}


# ---------------------------------------------------------------------------
# Defensive parsing at the untrusted subprocess boundary
# ---------------------------------------------------------------------------


async def test_envelope_non_dict_json_errored(monkeypatch):
    """`json.loads(stdout)` returning an array must not crash the provider."""
    proc = _FakeProc(stdout=json.dumps([1, 2, 3]).encode())
    _install_fake_proc(monkeypatch, proc)

    result = await ClaudeCliProvider().review(_make_request())

    assert result.disposition == "errored"
    assert result.error_category == "malformed_envelope"
    assert "not an object" in result.error


async def test_result_payload_non_dict_errored(monkeypatch):
    """`result` field carries a JSON value that isn't an object."""
    envelope = dict(SUCCESS_ENVELOPE)
    envelope["result"] = json.dumps(["not", "a", "review"])
    proc = _FakeProc(stdout=json.dumps(envelope).encode())
    _install_fake_proc(monkeypatch, proc)

    result = await ClaudeCliProvider().review(_make_request())

    assert result.disposition == "errored"
    assert result.error_category == "malformed_envelope"
    assert "not an object" in result.error


async def test_findings_filters_non_dict_items(monkeypatch):
    """Non-object items in the findings array must be skipped, not raised."""
    envelope = dict(SUCCESS_ENVELOPE)
    envelope["result"] = json.dumps(
        {
            "summary": "mixed",
            "findings": [
                {"severity": "nit", "title": "keep me", "body": "ok"},
                "rogue string",
                None,
                42,
                {"severity": "comment", "title": "also keep", "body": "ok"},
            ],
        }
    )
    proc = _FakeProc(stdout=json.dumps(envelope).encode())
    _install_fake_proc(monkeypatch, proc)

    result = await ClaudeCliProvider().review(_make_request())

    assert result.disposition == "posted"
    assert [f.title for f in result.findings] == ["keep me", "also keep"]


async def test_usage_non_dict_safely_zeroed(monkeypatch):
    """`usage` of the wrong type must not crash; token fields fall back to 0."""
    envelope = dict(SUCCESS_ENVELOPE)
    envelope["usage"] = "not-a-dict"
    proc = _FakeProc(stdout=json.dumps(envelope).encode())
    _install_fake_proc(monkeypatch, proc)

    result = await ClaudeCliProvider().review(_make_request())

    assert result.disposition == "posted"
    assert result.usage is not None
    assert result.usage.input_tokens == 0
    assert result.usage.output_tokens == 0
    assert result.usage.cache_read_tokens == 0
    assert result.usage.cache_creation_tokens == 0


async def test_usage_string_numbers_coerced_safely(monkeypatch):
    """Token values arriving as strings like '12.0' must coerce, not raise."""
    envelope = dict(SUCCESS_ENVELOPE)
    envelope["usage"] = {
        "input_tokens": "10",
        "output_tokens": "12.0",
        "cache_read_input_tokens": "nonsense",
        "cache_creation_input_tokens": None,
    }
    proc = _FakeProc(stdout=json.dumps(envelope).encode())
    _install_fake_proc(monkeypatch, proc)

    result = await ClaudeCliProvider().review(_make_request())

    assert result.disposition == "posted"
    assert result.usage is not None
    assert result.usage.input_tokens == 10
    assert result.usage.output_tokens == 12
    assert result.usage.cache_read_tokens == 0
    assert result.usage.cache_creation_tokens == 0


async def test_model_usage_non_dict_stats_entries_tolerated(monkeypatch):
    """A non-dict entry in `modelUsage` must not raise; only valid entries score."""
    envelope = dict(SUCCESS_ENVELOPE)
    envelope["modelUsage"] = {
        "big-model": {"costUSD": 2.0},
        "bogus": "not-a-dict",
        "empty": None,
        "medium-model": {"costUSD": 1.0},
    }
    proc = _FakeProc(stdout=json.dumps(envelope).encode())
    _install_fake_proc(monkeypatch, proc)

    result = await ClaudeCliProvider().review(_make_request())

    assert result.disposition == "posted"
    assert result.model == "big-model"


async def test_total_cost_usd_non_numeric_defaults_to_zero(monkeypatch):
    """`total_cost_usd` of the wrong type must not crash the usage record."""
    envelope = dict(SUCCESS_ENVELOPE)
    envelope["total_cost_usd"] = "nonsense"
    proc = _FakeProc(stdout=json.dumps(envelope).encode())
    _install_fake_proc(monkeypatch, proc)

    result = await ClaudeCliProvider().review(_make_request())

    assert result.disposition == "posted"
    assert result.usage is not None
    assert result.usage.estimated_api_cost_usd == 0.0


# ---------------------------------------------------------------------------
# healthcheck — startup auth pre-flight
# ---------------------------------------------------------------------------


async def test_healthcheck_logged_in_returns_quietly(monkeypatch):
    proc = _FakeProc(
        stdout=json.dumps(
            {
                "loggedIn": True,
                "authMethod": "claude.ai",
                "apiProvider": "firstParty",
                "subscriptionType": "max",
            }
        ).encode()
    )
    calls = _install_fake_proc(monkeypatch, proc)

    await ClaudeCliProvider().healthcheck()

    assert len(calls) == 1
    argv = calls[0]
    assert argv[0] == "claude"
    assert "auth" in argv
    assert "status" in argv
    assert "--json" in argv


async def test_healthcheck_logged_out_raises_auth_error(monkeypatch):
    proc = _FakeProc(
        stdout=json.dumps({"loggedIn": False}).encode()
    )
    _install_fake_proc(monkeypatch, proc)

    with pytest.raises(ClaudeCliAuthError) as excinfo:
        await ClaudeCliProvider().healthcheck()

    assert "not authenticated" in str(excinfo.value)
    assert "claude setup-token" in str(excinfo.value)


async def test_healthcheck_missing_binary_raises_filenotfound(monkeypatch):
    _install_missing_binary(monkeypatch)
    provider = ClaudeCliProvider(
        ClaudeCliProviderConfig(binary="claude-does-not-exist")
    )

    with pytest.raises(FileNotFoundError):
        await provider.healthcheck()


async def test_healthcheck_non_zero_exit_raises_runtime(monkeypatch):
    proc = _FakeProc(stdout=b"", stderr=b"boom", returncode=3)
    _install_fake_proc(monkeypatch, proc)

    with pytest.raises(RuntimeError) as excinfo:
        await ClaudeCliProvider().healthcheck()

    assert "exited with 3" in str(excinfo.value)
    # must not be the auth-specific subclass
    assert not isinstance(excinfo.value, ClaudeCliAuthError)


async def test_healthcheck_malformed_output_raises_runtime(monkeypatch):
    proc = _FakeProc(stdout=b"not json")
    _install_fake_proc(monkeypatch, proc)

    with pytest.raises(RuntimeError) as excinfo:
        await ClaudeCliProvider().healthcheck()

    assert "non-JSON" in str(excinfo.value)


async def test_claude_cli_auth_error_is_runtime_error():
    """Callers can catch RuntimeError to handle any healthcheck failure."""
    assert issubclass(ClaudeCliAuthError, RuntimeError)


async def test_request_model_overrides_config_default(monkeypatch):
    """Caller-supplied ``request.metadata['model']`` always wins."""
    proc = _FakeProc(stdout=json.dumps(SUCCESS_ENVELOPE).encode())
    calls = _install_fake_proc(monkeypatch, proc)
    config = ClaudeCliProviderConfig(default_model="claude-haiku-4-5")
    provider = ClaudeCliProvider(config)

    request = _make_request(metadata={"repo": "tolldog/example", "model": "opus"})
    await provider.review(request)

    argv = calls[0]
    assert "--model" in argv
    m_idx = argv.index("--model")
    assert argv[m_idx + 1] == "opus"


async def test_config_default_model_used_when_request_silent(monkeypatch):
    """Empty ``request.metadata['model']`` falls through to config default."""
    proc = _FakeProc(stdout=json.dumps(SUCCESS_ENVELOPE).encode())
    calls = _install_fake_proc(monkeypatch, proc)
    config = ClaudeCliProviderConfig(default_model="claude-opus-4-7")
    provider = ClaudeCliProvider(config)

    await provider.review(_make_request())  # no metadata['model']

    argv = calls[0]
    assert "--model" in argv
    m_idx = argv.index("--model")
    assert argv[m_idx + 1] == "claude-opus-4-7"


async def test_no_model_at_all_omits_flag(monkeypatch):
    """Both empty → omit ``--model`` so claude -p picks its own ambient default."""
    proc = _FakeProc(stdout=json.dumps(SUCCESS_ENVELOPE).encode())
    calls = _install_fake_proc(monkeypatch, proc)

    await ClaudeCliProvider().review(_make_request())  # default config = ""

    argv = calls[0]
    assert "--model" not in argv
