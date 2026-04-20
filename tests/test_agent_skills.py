"""Skill-surface tests for :class:`ReviewerAgent`.

Uses :class:`khonliang_bus.testing.AgentTestHarness` to dispatch
directly to ``@handler`` methods — no bus, no FastAPI, no real
Ollama / Claude CLI subprocesses. The fake :class:`_RecordingProvider`
captures the :class:`ReviewRequest` it was called with so tests can
assert routing + arg-forwarding end-to-end.
"""

from __future__ import annotations

from typing import Any

import pytest
from khonliang_bus.testing import AgentTestHarness

from khonliang_reviewer import (
    ReviewFinding,
    ReviewProvider,
    ReviewRequest,
    ReviewResult,
    UsageEvent,
)
from reviewer.agent import ReviewerAgent
from reviewer.selector import ProviderSelector, SelectorConfig


class _RecordingProvider(ReviewProvider):
    """In-process fake provider. Records the last request it saw."""

    def __init__(self, name: str, response: ReviewResult):
        self.name = name
        self._response = response
        self.last_request: ReviewRequest | None = None

    async def review(self, request: ReviewRequest) -> ReviewResult:
        self.last_request = request
        return self._response


def _make_result(
    *,
    backend: str = "fake",
    model: str = "fake-model",
    disposition: str = "posted",
    findings: list[ReviewFinding] | None = None,
) -> ReviewResult:
    return ReviewResult(
        request_id="test-req",
        summary="ok",
        findings=findings or [],
        disposition=disposition,  # type: ignore[arg-type]
        backend=backend,
        model=model,
        usage=UsageEvent(
            timestamp=1.0,
            backend=backend,
            model=model,
            input_tokens=10,
            output_tokens=5,
        ),
    )


def _make_harness(
    providers: dict[str, ReviewProvider] | None = None,
    *,
    default_backend: str = "ollama",
    default_model: str = "qwen3.5",
) -> AgentTestHarness:
    """Build an AgentTestHarness with an injected :class:`ProviderSelector`.

    The default provider map registers a single fake under ``"ollama"``
    so the rule table's default fallback (``ollama`` / ``qwen3.5``)
    resolves cleanly in tests that don't care about provider identity.
    Tests that want multiple providers pass their own map; tests that
    want caller-override pass ``backend=...`` explicitly.
    """
    if providers is None:
        providers = {"ollama": _RecordingProvider("ollama", _make_result(backend="ollama", model="qwen3.5"))}
    selector = ProviderSelector(
        providers,
        SelectorConfig(
            default_backend=default_backend, default_model=default_model
        ),
    )
    return AgentTestHarness(ReviewerAgent, selector=selector)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_expected_skills_registered():
    harness = _make_harness()
    assert "review_text" in harness.skill_names
    assert "review_diff" in harness.skill_names


def test_skills_parameters_match_public_contract():
    harness = _make_harness()
    skill = next(s for s in harness.skills if s.name == "review_text")
    # contract: kind + content are required, everything else optional
    assert skill.parameters["kind"]["required"] is True
    assert skill.parameters["content"]["required"] is True
    for optional in ("instructions", "context", "backend", "model", "request_id", "metadata"):
        assert optional in skill.parameters
        assert skill.parameters[optional].get("required", False) is False


# ---------------------------------------------------------------------------
# review_text happy paths
# ---------------------------------------------------------------------------


async def test_review_text_routes_to_rule_table_default_backend():
    """Small content + pr_diff → rule table picks ollama/qwen3.5 (fallback)."""
    fake = _RecordingProvider("ollama", _make_result(backend="ollama", model="qwen3.5"))
    harness = _make_harness({"ollama": fake})

    result = await harness.call(
        "review_text",
        {"kind": "pr_diff", "content": "diff body"},
    )

    assert result["disposition"] == "posted"
    assert result["summary"] == "ok"
    assert fake.last_request is not None
    assert fake.last_request.kind == "pr_diff"
    assert fake.last_request.content == "diff body"
    assert fake.last_request.metadata["model"] == "qwen3.5"


async def test_review_text_caller_backend_override_picks_specific_provider():
    a = _RecordingProvider("a", _make_result(backend="a"))
    b = _RecordingProvider("b", _make_result(backend="b"))
    harness = _make_harness({"a": a, "b": b}, default_backend="a")

    await harness.call(
        "review_text",
        {"kind": "pr_diff", "content": "x", "backend": "b"},
    )

    assert a.last_request is None
    assert b.last_request is not None


async def test_review_text_caller_model_override_threads_to_metadata():
    """Caller-supplied model forces caller-override path and lands in metadata."""
    fake = _RecordingProvider("ollama", _make_result())
    harness = _make_harness({"ollama": fake})

    await harness.call(
        "review_text",
        {"kind": "pr_diff", "content": "x", "model": "kimi-k2.5:cloud"},
    )

    assert fake.last_request is not None
    assert fake.last_request.metadata["model"] == "kimi-k2.5:cloud"


async def test_review_text_forwards_instructions_and_context():
    fake = _RecordingProvider("ollama", _make_result())
    harness = _make_harness({"ollama": fake})

    await harness.call(
        "review_text",
        {
            "kind": "spec",
            "content": "spec body",
            "instructions": "prioritize correctness",
            "context": {"repo_profile": "python-async"},
        },
    )

    assert fake.last_request is not None
    assert fake.last_request.instructions == "prioritize correctness"
    assert fake.last_request.context == {"repo_profile": "python-async"}


async def test_review_text_generates_request_id_when_missing():
    fake = _RecordingProvider("ollama", _make_result())
    harness = _make_harness({"ollama": fake})

    await harness.call("review_text", {"kind": "pr_diff", "content": "x"})

    assert fake.last_request is not None
    assert fake.last_request.request_id.startswith("rev-")


async def test_review_text_honors_caller_request_id():
    fake = _RecordingProvider("ollama", _make_result())
    harness = _make_harness({"ollama": fake})

    await harness.call(
        "review_text",
        {"kind": "pr_diff", "content": "x", "request_id": "custom-id-123"},
    )

    assert fake.last_request is not None
    assert fake.last_request.request_id == "custom-id-123"


async def test_review_text_merges_caller_metadata_with_model():
    fake = _RecordingProvider("ollama", _make_result())
    harness = _make_harness({"ollama": fake})

    await harness.call(
        "review_text",
        {
            "kind": "pr_diff",
            "content": "x",
            "metadata": {"repo": "tolldog/x", "pr_number": 7},
        },
    )

    assert fake.last_request is not None
    # caller metadata preserved
    assert fake.last_request.metadata["repo"] == "tolldog/x"
    assert fake.last_request.metadata["pr_number"] == 7
    # rule-table-chosen model injected alongside
    assert fake.last_request.metadata["model"] == "qwen3.5"


# ---------------------------------------------------------------------------
# review_text error paths
# ---------------------------------------------------------------------------


async def test_review_text_missing_kind_returns_error():
    harness = _make_harness()
    result = await harness.call("review_text", {"content": "x"})
    assert "error" in result
    assert "kind" in result["error"]


async def test_review_text_missing_content_returns_error():
    harness = _make_harness()
    result = await harness.call("review_text", {"kind": "pr_diff"})
    assert "error" in result
    assert "content" in result["error"]


async def test_review_text_content_not_string_returns_error():
    harness = _make_harness()
    result = await harness.call("review_text", {"kind": "pr_diff", "content": 42})
    assert "error" in result


async def test_review_text_unknown_backend_returns_error():
    fake = _RecordingProvider("ollama", _make_result())
    harness = _make_harness({"ollama": fake})
    result = await harness.call(
        "review_text",
        {"kind": "pr_diff", "content": "x", "backend": "nope"},
    )
    assert "error" in result
    assert "nope" in result["error"]
    assert fake.last_request is None


# ---------------------------------------------------------------------------
# review_diff
# ---------------------------------------------------------------------------


async def test_review_diff_sets_kind_and_content_from_diff():
    fake = _RecordingProvider("ollama", _make_result())
    harness = _make_harness({"ollama": fake})

    await harness.call("review_diff", {"diff": "diff --git a/x b/x"})

    assert fake.last_request is not None
    assert fake.last_request.kind == "pr_diff"
    assert fake.last_request.content == "diff --git a/x b/x"


async def test_review_diff_forwards_other_kwargs():
    fake = _RecordingProvider("ollama", _make_result())
    harness = _make_harness({"ollama": fake})

    await harness.call(
        "review_diff",
        {
            "diff": "x",
            "instructions": "careful",
            "context": {"k": "v"},
            "backend": "ollama",
            "model": "custom-model",
        },
    )

    req = fake.last_request
    assert req is not None
    assert req.instructions == "careful"
    assert req.context == {"k": "v"}
    assert req.metadata["model"] == "custom-model"


async def test_review_diff_missing_diff_returns_error():
    fake = _RecordingProvider("ollama", _make_result())
    harness = _make_harness({"ollama": fake})
    result = await harness.call("review_diff", {})
    assert "error" in result
    assert fake.last_request is None


async def test_review_diff_diff_not_string_returns_error():
    fake = _RecordingProvider("ollama", _make_result())
    harness = _make_harness({"ollama": fake})
    result = await harness.call("review_diff", {"diff": 123})
    assert "error" in result


# ---------------------------------------------------------------------------
# Rule-table integration (no caller override → policy picks)
# ---------------------------------------------------------------------------


async def test_rule_table_routes_docs_kind_to_ollama():
    """A spec / doc / fr review (no override) → ollama + qwen3.5 per rule table."""
    ollama = _RecordingProvider("ollama", _make_result(backend="ollama"))
    claude = _RecordingProvider("claude_cli", _make_result(backend="claude_cli"))
    harness = _make_harness({"ollama": ollama, "claude_cli": claude})

    await harness.call(
        "review_text",
        {"kind": "spec", "content": "# A small spec\n\nContent."},
    )

    assert ollama.last_request is not None
    assert claude.last_request is None
    assert ollama.last_request.metadata["model"] == "qwen3.5"


async def test_rule_table_routes_large_diff_to_claude():
    """≥2000 lines (or ≥20 files) → claude_cli per rule table."""
    ollama = _RecordingProvider("ollama", _make_result(backend="ollama"))
    claude = _RecordingProvider("claude_cli", _make_result(backend="claude_cli", model="claude"))
    harness = _make_harness({"ollama": ollama, "claude_cli": claude})

    big_diff = "diff --git a/f b/f\n" + ("+line\n" * 2500)
    await harness.call("review_diff", {"diff": big_diff})

    assert ollama.last_request is None
    assert claude.last_request is not None
    assert claude.last_request.metadata["model"] == "claude"


async def test_rule_table_long_context_routes_to_kimi():
    """≥5000 lines AND ≥10 files → ollama + kimi-k2.5:cloud per rule table."""
    ollama = _RecordingProvider("ollama", _make_result(backend="ollama"))
    claude = _RecordingProvider("claude_cli", _make_result(backend="claude_cli"))
    harness = _make_harness({"ollama": ollama, "claude_cli": claude})

    # 12 file headers × ~500 lines each = ~6000 lines across 12 files
    per_file = "diff --git a/x b/x\n" + ("+line\n" * 500)
    big_multi_file_diff = per_file * 12
    await harness.call("review_diff", {"diff": big_multi_file_diff})

    assert ollama.last_request is not None
    assert claude.last_request is None
    assert ollama.last_request.metadata["model"] == "kimi-k2.5:cloud"


async def test_caller_override_bypasses_rule_table():
    """When caller specifies backend, rule table is NOT consulted."""
    ollama = _RecordingProvider("ollama", _make_result(backend="ollama"))
    claude = _RecordingProvider("claude_cli", _make_result(backend="claude_cli"))
    harness = _make_harness({"ollama": ollama, "claude_cli": claude})

    # Large diff would route to claude via rule table, but caller says ollama
    big_diff = "diff --git a/f b/f\n" + ("+line\n" * 3000)
    await harness.call(
        "review_diff",
        {"diff": big_diff, "backend": "ollama", "model": "qwen3.5"},
    )

    assert ollama.last_request is not None
    assert claude.last_request is None


async def test_context_diff_size_overrides_content_estimate():
    """Caller-authoritative counts in context win over content estimation."""
    ollama = _RecordingProvider("ollama", _make_result(backend="ollama"))
    claude = _RecordingProvider("claude_cli", _make_result(backend="claude_cli", model="claude"))
    harness = _make_harness({"ollama": ollama, "claude_cli": claude})

    # Tiny inline content — but context says it's actually a 5000-line, 15-file
    # diff (maybe the caller passed a summary instead of the full body). Rule
    # table should route based on the authoritative counts, not the estimate.
    await harness.call(
        "review_text",
        {
            "kind": "pr_diff",
            "content": "summary of large diff",
            "context": {"diff_line_count": 6000, "diff_file_count": 15},
        },
    )

    assert ollama.last_request is not None
    assert ollama.last_request.metadata["model"] == "kimi-k2.5:cloud"


# ---------------------------------------------------------------------------
# Default selector construction (without injection)
# ---------------------------------------------------------------------------


def test_default_selector_constructs_both_providers_from_empty_config(tmp_path):
    """Without a config file, defaults are used; both providers present."""
    agent = ReviewerAgent(
        agent_id="reviewer-test",
        bus_url="http://mock",
        config_path="",
    )
    selector = agent._ensure_selector()
    assert set(selector.providers) == {"claude_cli", "ollama"}
    assert selector.config.default_backend == "ollama"
    assert selector.config.default_model == "qwen3.5"


def test_default_selector_honors_config_yaml(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "default_provider: claude_cli\n"
        "default_model: claude-opus-4-7\n"
        "providers:\n"
        "  ollama:\n"
        "    base_url: http://example:11434/v1\n"
    )
    agent = ReviewerAgent(
        agent_id="reviewer-test",
        bus_url="http://mock",
        config_path=str(config_path),
    )
    selector = agent._ensure_selector()
    assert selector.config.default_backend == "claude_cli"
    assert selector.config.default_model == "claude-opus-4-7"
