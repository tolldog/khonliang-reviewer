"""Tests for the rule-table policy."""

from __future__ import annotations

from reviewer.rules import (
    CTX_LARGE,
    CTX_MEDIUM,
    CTX_SMALL,
    DEFAULT_FALLBACK,
    DEFAULT_RULES,
    PolicyDecision,
    PolicyInput,
    Rule,
    decide,
)


# ---------------------------------------------------------------------------
# Default-rules coverage
# ---------------------------------------------------------------------------


def test_empty_input_hits_fallback():
    decision = decide(PolicyInput())
    assert decision == DEFAULT_FALLBACK
    assert decision.backend == "ollama"
    assert decision.model == "qwen2.5-coder:14b"


def test_small_code_diff_hits_fallback():
    decision = decide(PolicyInput(kind="pr_diff", diff_line_count=50, diff_file_count=3))
    assert decision == DEFAULT_FALLBACK


def test_docs_kind_routes_to_qwen_small():
    for kind in ("spec", "doc", "fr", "pr_description"):
        decision = decide(PolicyInput(kind=kind, diff_line_count=20))
        assert decision.backend == "ollama"
        assert decision.model == "qwen2.5-coder:14b"
        assert decision.context_window_floor == CTX_SMALL
        assert "text-kind" in decision.reason


def test_large_diff_routes_to_claude():
    decision = decide(
        PolicyInput(kind="pr_diff", diff_line_count=2500, diff_file_count=5)
    )
    assert decision.backend == "claude_cli"
    assert decision.model == "claude"
    assert decision.context_window_floor == CTX_MEDIUM


def test_many_files_routes_to_claude():
    """Architectural-scope by file-count alone, not just line-count."""
    decision = decide(
        PolicyInput(kind="pr_diff", diff_line_count=200, diff_file_count=25)
    )
    assert decision.backend == "claude_cli"
    assert decision.model == "claude"


def test_very_large_diff_routes_to_long_context():
    decision = decide(
        PolicyInput(kind="pr_diff", diff_line_count=6000, diff_file_count=15)
    )
    assert decision.backend == "ollama"
    assert decision.model == "kimi-k2.5:cloud"
    assert decision.context_window_floor == CTX_LARGE


def test_long_context_rule_beats_large_diff_rule():
    """Order matters — long-context predicate must match before large-diff."""
    # This input satisfies both _long_context_diff AND _large_diff; the
    # long-context rule must win because it appears first.
    decision = decide(
        PolicyInput(kind="pr_diff", diff_line_count=8000, diff_file_count=40)
    )
    assert decision.model == "kimi-k2.5:cloud"


def test_large_diff_with_few_files_stays_on_claude():
    """≥2000 lines but only a couple of files — still architectural, Claude wins."""
    decision = decide(
        PolicyInput(kind="pr_diff", diff_line_count=3000, diff_file_count=2)
    )
    assert decision.backend == "claude_cli"


def test_docs_kind_with_huge_diff_still_goes_claude_or_long_context():
    """Kind routing is NOT terminal — a 10k-line spec still needs long-context."""
    decision = decide(
        PolicyInput(kind="spec", diff_line_count=6000, diff_file_count=12)
    )
    # Either claude (large-diff) or kimi (long-context) is acceptable here,
    # depending on future rule order. Lock the current behavior: long-context
    # rule evaluates first and matches, so kimi wins.
    assert decision.model == "kimi-k2.5:cloud"


# ---------------------------------------------------------------------------
# Rule-table plumbing
# ---------------------------------------------------------------------------


def test_empty_rules_list_always_returns_fallback():
    decision = decide(
        PolicyInput(kind="pr_diff", diff_line_count=10_000, diff_file_count=50),
        rules=[],
    )
    assert decision == DEFAULT_FALLBACK


def test_custom_rules_take_precedence():
    custom_decision = PolicyDecision(
        backend="ollama",
        model="glm-4.7-flash",
        context_window_floor=CTX_MEDIUM,
        reason="custom",
    )
    custom_rules = [
        Rule(
            predicate=lambda inp: inp.kind == "pr_diff",
            decision=custom_decision,
            name="always_glm_for_pr_diff",
        ),
    ]
    decision = decide(PolicyInput(kind="pr_diff"), rules=custom_rules)
    assert decision == custom_decision


def test_broken_predicate_is_skipped_not_raised():
    """A faulty rule must not block the rest of the table from evaluating."""
    def explode(_inp: PolicyInput) -> bool:
        raise RuntimeError("boom")

    custom_rules = [
        Rule(predicate=explode, decision=DEFAULT_FALLBACK, name="broken"),
        Rule(
            predicate=lambda inp: inp.kind == "pr_diff",
            decision=PolicyDecision(
                backend="ollama",
                model="working",
                context_window_floor=CTX_SMALL,
                reason="fallback rule after broken one",
            ),
            name="working",
        ),
    ]
    decision = decide(PolicyInput(kind="pr_diff"), rules=custom_rules)
    assert decision.model == "working"


def test_decisions_are_explainable():
    """Every default decision must carry a non-empty ``reason``."""
    for rule in DEFAULT_RULES:
        assert rule.decision.reason.strip() != ""
    assert DEFAULT_FALLBACK.reason.strip() != ""


def test_all_default_rule_decisions_have_valid_backends():
    for rule in DEFAULT_RULES:
        assert rule.decision.backend in {"ollama", "claude_cli"}
    assert DEFAULT_FALLBACK.backend in {"ollama", "claude_cli"}


def test_context_window_constants_are_ascending():
    assert CTX_SMALL < CTX_MEDIUM < CTX_LARGE
