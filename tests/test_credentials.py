"""Tests for :mod:`reviewer.credentials` — GitHub token discovery.

Fixture tokens in this module are obviously fake (``test-token-xxx``)
so nothing in the test suite could ever resemble a real credential
leaking through a commit.
"""

from __future__ import annotations

import subprocess
from typing import Any

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


# ---------------------------------------------------------------------------
# Subprocess env sanitization — whitespace-only vars stripped before gh
# ---------------------------------------------------------------------------


def test_blank_env_vars_are_stripped_before_gh_subprocess(monkeypatch):
    """gh auth token must NOT inherit a whitespace-only GITHUB_TOKEN.

    Otherwise gh treats the blank value as "authenticated with this
    token" and skips its keyring lookup, defeating the whitespace
    fallback this module promises.
    """
    monkeypatch.setenv("GITHUB_TOKEN", "  ")
    monkeypatch.setenv("GH_TOKEN", "\n\t ")
    monkeypatch.setenv("SOME_OTHER_VAR", "keep-me")
    captured_env: dict[str, str] = {}

    def fake_run(cmd, **kwargs: Any):
        captured_env.update(kwargs.get("env") or {})
        return subprocess.CompletedProcess(cmd, 0, stdout="test-token-keyring\n", stderr="")

    monkeypatch.setattr(credentials.subprocess, "run", fake_run)

    assert get_github_token() == "test-token-keyring"
    # The two github-token env vars must be gone from the child env...
    assert "GITHUB_TOKEN" not in captured_env
    assert "GH_TOKEN" not in captured_env
    # ...but the rest of the parent env passes through unchanged
    assert captured_env.get("SOME_OTHER_VAR") == "keep-me"


def test_unrelated_env_vars_preserved_in_subprocess(monkeypatch):
    """Only blank GITHUB_TOKEN / GH_TOKEN are stripped; everything else stays."""
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.setenv("UNRELATED", "preserve-me")
    captured_env: dict[str, str] = {}

    def fake_run(cmd, **kwargs: Any):
        captured_env.update(kwargs.get("env") or {})
        return subprocess.CompletedProcess(cmd, 0, stdout="x\n", stderr="")

    monkeypatch.setattr(credentials.subprocess, "run", fake_run)
    get_github_token()
    assert captured_env.get("UNRELATED") == "preserve-me"


# ---------------------------------------------------------------------------
# get_tabbyapi_key (fr_khonliang-reviewer_0e7ccff1)
# ---------------------------------------------------------------------------

from reviewer.credentials import get_tabbyapi_key  # noqa: E402


def test_tabbyapi_env_var_wins(monkeypatch, tmp_path):
    tokens = tmp_path / "api_tokens.yml"
    tokens.write_text("api_key: from-file\nadmin_key: admin-secret\n")
    monkeypatch.setenv("TABBY_API_KEY", "from-env")
    monkeypatch.setenv("TABBY_API_TOKENS_FILE", str(tokens))
    assert get_tabbyapi_key() == "from-env"


def test_tabbyapi_falls_back_to_tokens_file(monkeypatch, tmp_path):
    tokens = tmp_path / "api_tokens.yml"
    tokens.write_text("admin_key: admin-secret\napi_key: 'file-key'\n")
    monkeypatch.delenv("TABBY_API_KEY", raising=False)
    monkeypatch.setenv("TABBY_API_TOKENS_FILE", str(tokens))
    # api_key is read (quotes stripped); admin_key is NEVER returned —
    # the reviewer has no business with admin endpoints.
    assert get_tabbyapi_key() == "file-key"


def test_tabbyapi_whitespace_env_falls_through(monkeypatch, tmp_path):
    tokens = tmp_path / "api_tokens.yml"
    tokens.write_text("api_key: file-key\n")
    monkeypatch.setenv("TABBY_API_KEY", "   ")
    monkeypatch.setenv("TABBY_API_TOKENS_FILE", str(tokens))
    assert get_tabbyapi_key() == "file-key"


def test_tabbyapi_missing_file_returns_none(monkeypatch, tmp_path):
    monkeypatch.delenv("TABBY_API_KEY", raising=False)
    monkeypatch.setenv("TABBY_API_TOKENS_FILE", str(tmp_path / "nope.yml"))
    assert get_tabbyapi_key() is None


def test_tabbyapi_file_without_api_key_returns_none(monkeypatch, tmp_path):
    tokens = tmp_path / "api_tokens.yml"
    tokens.write_text("admin_key: admin-secret\n")
    monkeypatch.delenv("TABBY_API_KEY", raising=False)
    monkeypatch.setenv("TABBY_API_TOKENS_FILE", str(tokens))
    assert get_tabbyapi_key() is None
