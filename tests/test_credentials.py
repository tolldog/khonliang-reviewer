"""Tests for :mod:`reviewer.credentials` — GitHub token discovery.

Fixture tokens in this module are obviously fake (``test-token-xxx``)
so nothing in the test suite could ever resemble a real credential
leaking through a commit.
"""

from __future__ import annotations

import subprocess
from typing import Any

import pytest

from reviewer import credentials
from reviewer.credentials import get_github_token


# ---------------------------------------------------------------------------
# env-var chain
# ---------------------------------------------------------------------------


def test_github_token_env_var_wins(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "test-token-env")
    monkeypatch.delenv("GH_TOKEN", raising=False)

    assert get_github_token() == "test-token-env"


def test_gh_token_fallback_when_github_token_missing(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.setenv("GH_TOKEN", "test-token-gh")

    assert get_github_token() == "test-token-gh"


def test_github_token_strips_whitespace(monkeypatch):
    """Trailing newlines from ``export FOO=$(cmd)`` must not pollute the token."""
    monkeypatch.setenv("GITHUB_TOKEN", "  test-token-ws  \n")
    monkeypatch.delenv("GH_TOKEN", raising=False)

    assert get_github_token() == "test-token-ws"


def test_empty_env_falls_through_to_subprocess(monkeypatch):
    """Empty/blank env var must NOT short-circuit as a valid token."""
    monkeypatch.setenv("GITHUB_TOKEN", "")
    monkeypatch.setenv("GH_TOKEN", "   ")
    captured: list[tuple[str, ...]] = []

    def fake_run(cmd, **_: Any):
        captured.append(tuple(cmd))
        return subprocess.CompletedProcess(cmd, 0, stdout="test-token-gh-cli\n", stderr="")

    monkeypatch.setattr(credentials.subprocess, "run", fake_run)

    assert get_github_token() == "test-token-gh-cli"
    assert captured  # subprocess actually reached


# ---------------------------------------------------------------------------
# gh subprocess fallback
# ---------------------------------------------------------------------------


def test_gh_auth_token_subprocess_returns_token(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)

    def fake_run(cmd, **_: Any):
        assert cmd == ["gh", "auth", "token", "--hostname", "github.com"]
        return subprocess.CompletedProcess(cmd, 0, stdout="test-token-keyring\n", stderr="")

    monkeypatch.setattr(credentials.subprocess, "run", fake_run)

    assert get_github_token() == "test-token-keyring"


def test_gh_binary_missing_returns_none(monkeypatch):
    """gh not installed is a normal case — fall back to None, not a crash."""
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)

    def fake_run(cmd, **_: Any):
        raise FileNotFoundError(2, "No such file or directory: 'gh'")

    monkeypatch.setattr(credentials.subprocess, "run", fake_run)

    assert get_github_token() is None


def test_gh_logged_out_returns_none(monkeypatch):
    """Non-zero exit from gh (operator logged out) yields None, no exception."""
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)

    def fake_run(cmd, **_: Any):
        return subprocess.CompletedProcess(
            cmd, 1, stdout="", stderr="not logged in"
        )

    monkeypatch.setattr(credentials.subprocess, "run", fake_run)

    assert get_github_token() is None


def test_gh_timeout_returns_none(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)

    def fake_run(cmd, **_: Any):
        raise subprocess.TimeoutExpired(cmd, 10)

    monkeypatch.setattr(credentials.subprocess, "run", fake_run)

    assert get_github_token() is None


def test_gh_os_error_returns_none(monkeypatch):
    """Generic OSError (permission denied, bad executable) returns None."""
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)

    def fake_run(cmd, **_: Any):
        raise OSError(13, "Permission denied")

    monkeypatch.setattr(credentials.subprocess, "run", fake_run)

    assert get_github_token() is None


def test_gh_captures_stdout_stderr_to_prevent_token_leak(monkeypatch):
    """The subprocess must run with capture enabled so stdout is not echoed."""
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    captured_kwargs: dict[str, Any] = {}

    def fake_run(cmd, **kwargs: Any):
        captured_kwargs.update(kwargs)
        return subprocess.CompletedProcess(cmd, 0, stdout="test-token-capture\n", stderr="")

    monkeypatch.setattr(credentials.subprocess, "run", fake_run)
    get_github_token()

    assert captured_kwargs.get("capture_output") is True
    assert captured_kwargs.get("text") is True
    # `check=False` so non-zero exit doesn't raise and leak stderr via traceback
    assert captured_kwargs.get("check") is False


# ---------------------------------------------------------------------------
# No-caching guarantee
# ---------------------------------------------------------------------------


def test_rotation_picked_up_on_each_call(monkeypatch):
    """A token change between calls must be visible immediately."""
    monkeypatch.setenv("GITHUB_TOKEN", "rotated-v1")
    assert get_github_token() == "rotated-v1"
    monkeypatch.setenv("GITHUB_TOKEN", "rotated-v2")
    assert get_github_token() == "rotated-v2"
