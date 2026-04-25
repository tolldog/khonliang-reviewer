"""Tests for ``reviewer.registry`` — ProviderRegistry + cheap availability probe.

The registry is read-mostly: tests construct one, register one or more
fake providers, and assert the resulting :class:`ProviderRegistration`
shape. The cheap-availability probe is mostly tested via monkeypatch
on ``shutil.which`` + the ``OPENAI_API_KEY`` env var so we don't
require a real ``codex`` / ``claude`` binary on the test host.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from khonliang_reviewer import ReviewProvider
from reviewer.registry import (
    ProviderRegistration,
    ProviderRegistry,
    _check_availability,
)


class _Fake(ReviewProvider):
    """Minimal stub provider for registry tests."""

    def __init__(self, name: str, *, config: object | None = None):
        self.name = name
        if config is not None:
            self.config = config

    async def review(self, request):  # pragma: no cover — not exercised
        raise NotImplementedError


# ---------------------------------------------------------------------------
# ProviderRegistry: register / list / providers
# ---------------------------------------------------------------------------


def test_register_and_list_returns_basic_shape():
    registry = ProviderRegistry()
    registry.register(
        _Fake("ollama"),
        default_model="qwen2.5-coder:14b",
        declared_models=["qwen2.5-coder:14b", "glm-4.7-flash"],
    )

    items = registry.list()
    assert len(items) == 1
    item = items[0]
    assert isinstance(item, ProviderRegistration)
    assert item.backend == "ollama"
    assert item.default_model == "qwen2.5-coder:14b"
    # Default model first; declared models appended (deduped).
    assert item.models == ("qwen2.5-coder:14b", "glm-4.7-flash")
    assert item.available is True
    assert item.reason == ""


def test_register_dedupes_declared_models():
    registry = ProviderRegistry()
    registry.register(
        _Fake("ollama"),
        default_model="",
        declared_models=["a", "b", "a", "c", "b"],
    )
    items = registry.list()
    # No default surfaced (empty), so models is just the deduped declared.
    assert items[0].models == ("a", "b", "c")


def test_default_model_first_then_declared_without_duplication():
    registry = ProviderRegistry()
    registry.register(
        _Fake("ollama"),
        default_model="qwen2.5-coder:14b",
        declared_models=[
            "glm-4.7-flash",
            "qwen2.5-coder:14b",  # already the default — must not appear twice
            "kimi-k2.5:cloud",
        ],
    )
    items = registry.list()
    # Default first, then any declared that isn't the default — same
    # contract used by ``list_models`` consumers.
    assert items[0].models == (
        "qwen2.5-coder:14b",
        "glm-4.7-flash",
        "kimi-k2.5:cloud",
    )


def test_register_empty_declared_surfaces_only_default():
    registry = ProviderRegistry()
    registry.register(_Fake("ollama"), default_model="qwen2.5-coder:14b")
    assert registry.list()[0].models == ("qwen2.5-coder:14b",)


def test_register_no_default_no_declared_yields_empty_models():
    registry = ProviderRegistry()
    registry.register(_Fake("ollama"))
    assert registry.list()[0].models == ()


def test_register_rejects_blank_provider_name():
    registry = ProviderRegistry()
    with pytest.raises(ValueError):
        registry.register(_Fake(""), default_model="anything")


def test_providers_property_is_a_mapping_for_selector():
    """``ProviderSelector`` consumes ``Mapping[str, ReviewProvider]`` —
    the registry's ``providers`` property must satisfy that contract."""
    a = _Fake("a")
    b = _Fake("b")
    registry = ProviderRegistry()
    registry.register(a)
    registry.register(b)
    providers = registry.providers
    assert set(providers.keys()) == {"a", "b"}
    assert providers["a"] is a
    assert providers["b"] is b


def test_register_overwrites_when_same_backend_re_registered():
    """Re-registering the same backend swaps the provider + metadata.

    Useful for tests that want to inject a fake over a default
    registration without having to construct a fresh registry.
    """
    registry = ProviderRegistry()
    first = _Fake("ollama")
    second = _Fake("ollama")
    registry.register(first, default_model="x", declared_models=["x"])
    registry.register(second, default_model="y", declared_models=["y", "z"])

    assert registry.providers["ollama"] is second
    item = registry.list()[0]
    assert item.default_model == "y"
    assert item.models == ("y", "z")


# ---------------------------------------------------------------------------
# Backend filter
# ---------------------------------------------------------------------------


def test_list_with_backend_filter_returns_only_match():
    registry = ProviderRegistry()
    registry.register(_Fake("ollama"))
    registry.register(_Fake("claude_cli"))
    registry.register(_Fake("codex_cli"))

    items = registry.list(backend="claude_cli")
    assert len(items) == 1
    assert items[0].backend == "claude_cli"


def test_list_with_unknown_backend_filter_returns_empty():
    registry = ProviderRegistry()
    registry.register(_Fake("ollama"))
    assert registry.list(backend="nonexistent") == []


# ---------------------------------------------------------------------------
# _check_availability: claude_cli
# ---------------------------------------------------------------------------


def test_check_availability_claude_cli_present(monkeypatch):
    monkeypatch.setattr(
        "reviewer.registry.shutil.which",
        lambda binary: "/usr/local/bin/claude" if binary == "claude" else None,
    )
    provider = _Fake("claude_cli", config=SimpleNamespace(binary="claude"))
    available, reason = _check_availability("claude_cli", provider)
    assert available is True
    assert reason == ""


def test_check_availability_claude_cli_missing(monkeypatch):
    monkeypatch.setattr("reviewer.registry.shutil.which", lambda _b: None)
    provider = _Fake("claude_cli", config=SimpleNamespace(binary="claude"))
    available, reason = _check_availability("claude_cli", provider)
    assert available is False
    assert "claude binary not found" in reason


# ---------------------------------------------------------------------------
# _check_availability: codex_cli
# ---------------------------------------------------------------------------


def test_check_availability_codex_cli_present_with_oauth(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "reviewer.registry.shutil.which",
        lambda binary: "/usr/local/bin/codex" if binary == "codex" else None,
    )
    fake_home = tmp_path / "home"
    auth_dir = fake_home / ".codex"
    auth_dir.mkdir(parents=True)
    (auth_dir / "auth.json").write_text("{}")
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    provider = _Fake("codex_cli", config=SimpleNamespace(binary="codex"))
    available, reason = _check_availability("codex_cli", provider)
    assert available is True
    assert reason == ""


def test_check_availability_codex_cli_present_with_env_key(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "reviewer.registry.shutil.which",
        lambda binary: "/usr/local/bin/codex" if binary == "codex" else None,
    )
    monkeypatch.setenv("HOME", str(tmp_path))  # no ~/.codex/auth.json
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")

    provider = _Fake("codex_cli", config=SimpleNamespace(binary="codex"))
    available, reason = _check_availability("codex_cli", provider)
    assert available is True


def test_check_availability_codex_cli_no_auth(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "reviewer.registry.shutil.which",
        lambda binary: "/usr/local/bin/codex" if binary == "codex" else None,
    )
    monkeypatch.setenv("HOME", str(tmp_path))  # no ~/.codex/auth.json
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    provider = _Fake("codex_cli", config=SimpleNamespace(binary="codex"))
    available, reason = _check_availability("codex_cli", provider)
    assert available is False
    assert "no codex auth" in reason


def test_check_availability_codex_cli_no_binary(monkeypatch):
    monkeypatch.setattr("reviewer.registry.shutil.which", lambda _b: None)
    provider = _Fake("codex_cli", config=SimpleNamespace(binary="codex"))
    available, reason = _check_availability("codex_cli", provider)
    assert available is False
    assert "codex binary not found" in reason


# ---------------------------------------------------------------------------
# _check_availability: ollama
# ---------------------------------------------------------------------------


def test_check_availability_ollama_always_true():
    """Ollama liveness is a network probe — explicit healthcheck only.

    The cheap-probe contract here is "registered = available" so
    list_models doesn't make synchronous HTTP calls per backend.
    """
    provider = _Fake("ollama", config=SimpleNamespace(base_url="http://x:11434"))
    available, reason = _check_availability("ollama", provider)
    assert available is True
    assert reason == ""


# ---------------------------------------------------------------------------
# _check_availability: gh_copilot (planned fourth backend; probe is registered)
# ---------------------------------------------------------------------------


def test_check_availability_gh_copilot_present_with_token(monkeypatch):
    monkeypatch.setattr(
        "reviewer.registry.shutil.which",
        lambda binary: "/usr/local/bin/copilot" if binary == "copilot" else None,
    )
    monkeypatch.setenv("GH_TOKEN", "ghp_fake")

    provider = _Fake("gh_copilot", config=SimpleNamespace(binary="copilot"))
    available, reason = _check_availability("gh_copilot", provider)
    assert available is True


def test_check_availability_gh_copilot_no_binary(monkeypatch):
    monkeypatch.setattr("reviewer.registry.shutil.which", lambda _b: None)
    provider = _Fake("gh_copilot", config=SimpleNamespace(binary="copilot"))
    available, reason = _check_availability("gh_copilot", provider)
    assert available is False
    assert "copilot binary not found" in reason


# ---------------------------------------------------------------------------
# Unknown backend
# ---------------------------------------------------------------------------


def test_check_availability_unknown_backend_assumed_available():
    """Registry trusts whoever registered the provider; no probe by name."""
    provider = _Fake("future_backend")
    available, reason = _check_availability("future_backend", provider)
    assert available is True
    assert reason == ""


# ---------------------------------------------------------------------------
# Registration ProviderRegistration.to_dict()
# ---------------------------------------------------------------------------


def test_registration_to_dict_preserves_models_as_list():
    """``models`` is a tuple internally for immutability; ``to_dict``
    converts to a list because the bus skill response shape requires
    JSON-serializable types and ``json.dumps`` doesn't round-trip
    tuples with full fidelity (becomes a list anyway)."""
    reg = ProviderRegistration(
        backend="ollama",
        default_model="qwen2.5-coder:14b",
        models=("qwen2.5-coder:14b", "glm-4.7-flash"),
        available=True,
    )
    d = reg.to_dict()
    assert d == {
        "backend": "ollama",
        "default_model": "qwen2.5-coder:14b",
        "models": ["qwen2.5-coder:14b", "glm-4.7-flash"],
        "available": True,
        "reason": "",
    }
    assert isinstance(d["models"], list)
