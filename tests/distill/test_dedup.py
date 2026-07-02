"""Tests for ``reviewer.distill.transforms.dedup.apply_dedup``.

Pins the contract every other transform-PR depends on:
- ``dedup="none"`` is identity (in and out are the same object).
- ``dedup="exact"`` (the default) collapses identical
  (title, body, path, line, section) findings.
- ``dedup="title_substring"`` collapses findings whose titles share
  a substring relationship (case-insensitive).
- The merged survivor's severity is bumped to the highest in the
  group; the survivor's other fields are the earlier finding's.
- Feature-preservation: a unique outlier never merges into a
  dissimilar finding (the "10×-outlier survives unchanged" test
  from MS-B's design principle).
- ``dedup="semantic"`` raises until that transform lands.
"""

from __future__ import annotations

import pytest
from khonliang_reviewer import ReviewFinding, ReviewResult

from reviewer.distill.transforms.dedup import apply_dedup
from reviewer.rules.distill import DistillConfig


def _result(*findings: ReviewFinding) -> ReviewResult:
    return ReviewResult(request_id="req-test", summary="ok", findings=list(findings))


def _f(
    severity: str,
    title: str,
    body: str = "b",
    path: str | None = None,
    line: int | None = None,
) -> ReviewFinding:
    return ReviewFinding(severity=severity, title=title, body=body, path=path, line=line)  # type: ignore[arg-type]


def test_none_is_identity():
    """The opt-out strategy passes the result through unchanged —
    callers that need the raw emission stream (e.g. measuring a
    model's duplicate-emission rate) see every repeat.
    """
    result = _result(_f("nit", "x"), _f("nit", "x"))  # would dedup if asked
    out = apply_dedup(result, DistillConfig(dedup="none"))
    assert out is result


def test_exact_collapses_identical_pairs():
    """Three findings with identical (title, body) collapse to one.
    Severity rises to the max in the group.
    """
    result = _result(
        _f("nit", "X", "b"),
        _f("comment", "X", "b"),
        _f("concern", "X", "b"),
    )
    out = apply_dedup(result, DistillConfig(dedup="exact"))
    assert len(out.findings) == 1
    assert out.findings[0].severity == "concern"
    assert out.findings[0].title == "X"


def test_exact_keeps_distinct_findings():
    """Distinct (title, body) tuples don't merge — same title with
    different bodies stays separate so a model emitting two findings
    against the same line for different reasons keeps both.
    """
    result = _result(
        _f("nit", "X", "body 1"),
        _f("nit", "X", "body 2"),
    )
    out = apply_dedup(result, DistillConfig(dedup="exact"))
    assert len(out.findings) == 2
    # No merging happened → identity preserved (composes cleanly with
    # subsequent transforms that may be inert too).
    assert out is result


def test_exact_keeps_same_text_at_different_locations():
    """Identical terse text anchored to different files/lines is two
    distinct observations, not a repeat — location is part of the
    exact-dup key. A model legitimately emits "Missing docstring"
    against three different functions; collapsing them would erase
    two real findings.
    """
    result = _result(
        _f("nit", "Missing docstring", "Add one.", path="a.py", line=10),
        _f("nit", "Missing docstring", "Add one.", path="b.py", line=10),
        _f("nit", "Missing docstring", "Add one.", path="a.py", line=42),
    )
    out = apply_dedup(result, DistillConfig(dedup="exact"))
    assert len(out.findings) == 3
    assert out is result


def test_exact_keeps_copy_carrying_a_suggestion():
    """A copy that differs in ``suggestion`` or ``category`` is not a
    pure repeat (PR #70 codex P1): merging it away would drop the
    concrete suggested fix from GitHub-comment rendering and from
    ``sign_off_trailer``'s actionability check. Both survive.
    """
    plain = _f("concern", "Race in handler", "Lock is taken too late.")
    with_fix = ReviewFinding(
        severity="concern",
        title="Race in handler",
        body="Lock is taken too late.",
        suggestion="with self._lock:\n    self._start(agent_id)",
    )
    other_category = ReviewFinding(
        severity="concern",
        title="Race in handler",
        body="Lock is taken too late.",
        category="concurrency",
    )
    out = apply_dedup(_result(plain, with_fix, other_category), DistillConfig(dedup="exact"))
    assert len(out.findings) == 3


def test_exact_is_the_default_and_records_drops():
    """The dog_fa0e1a48 shape: one diff, the same byte-identical
    descriptive nit emitted 5 times. The default config (no explicit
    ``dedup``) collapses them to one survivor and records the four
    collapsed copies on ``dropped_findings`` — the drop is auditable,
    never silent.
    """
    copies = [
        _f("nit", "Code Clarification", "The SKIP branch returns early.")
        for _ in range(5)
    ]
    result = _result(*copies)
    out = apply_dedup(result, DistillConfig())
    assert len(out.findings) == 1
    assert len(out.dropped_findings) == 4
    assert all(d.title == "Code Clarification" for d in out.dropped_findings)


def test_title_substring_merges_supersets():
    """Common case: a model emits 'Missing test' AND 'Missing test
    for handler' as two findings about the same underlying gap.
    Both forms (longer ⊇ shorter, shorter ⊆ longer) merge into the
    earlier one.
    """
    result = _result(
        _f("nit", "Missing test"),
        _f("comment", "Missing test for handler"),
    )
    out = apply_dedup(result, DistillConfig(dedup="title_substring"))
    assert len(out.findings) == 1
    # Earlier finding survives; severity bumps to the higher one.
    assert out.findings[0].title == "Missing test"
    assert out.findings[0].severity == "comment"


def test_title_substring_is_case_insensitive():
    result = _result(_f("nit", "missing TEST"), _f("comment", "Missing Test for handler"))
    out = apply_dedup(result, DistillConfig(dedup="title_substring"))
    assert len(out.findings) == 1
    assert out.findings[0].severity == "comment"


def test_title_substring_keeps_unrelated_titles():
    """Distinct titles don't merge. 'Race condition' and 'Off-by-one
    error' are unrelated concerns even though they share short
    English words; substring matching is character-level so neither
    is a substring of the other.
    """
    result = _result(
        _f("nit", "Race condition"),
        _f("nit", "Off-by-one error"),
    )
    out = apply_dedup(result, DistillConfig(dedup="title_substring"))
    assert len(out.findings) == 2
    assert out is result


def test_title_substring_skips_empty_titles():
    """Empty title would technically be a substring of every other
    title, which would aggressively merge every summary-level
    finding into the first empty-title row. Skip empties so they
    stay distinct.
    """
    result = _result(_f("nit", ""), _f("nit", "Real finding"))
    out = apply_dedup(result, DistillConfig(dedup="title_substring"))
    assert len(out.findings) == 2


def test_semantic_strategy_raises():
    """Reserved for a future embedding-similarity transform; raise
    loudly so a misconfigured rule fails at runtime instead of
    silently degrading to ``none`` (which would mask the bug).
    """
    result = _result(_f("nit", "x"), _f("nit", "x"))
    with pytest.raises(ValueError, match="semantic"):
        apply_dedup(result, DistillConfig(dedup="semantic"))


def test_outlier_concern_survives_among_nits():
    """The MS-B feature-preservation invariant: an outlier concern
    surrounded by 20 distinct nits is never merged into anything.
    Tests both supported strategies.

    Titles are chosen to be substring-distinct (no title contains
    another) so this test exercises the dedup logic on real-world-
    shaped inputs rather than ambiguous "Nit 1" / "Nit 10" strings
    that would (correctly) merge under the title_substring strategy.
    """
    nit_titles = [
        "Unused import os",
        "Trailing whitespace in module",
        "Variable naming style mismatch",
        "Docstring missing summary line",
        "Magic literal could be a constant",
        "Redundant parentheses around return",
        "Inconsistent quote style",
        "TODO comment lacks owner",
        "Type annotation could narrow",
        "F-string without interpolation",
        "Single-letter loop variable",
        "Comment is stale relative to code",
        "Mutable default argument warning",
        "Wildcard star-import discouraged",
        "Bare except swallows tracebacks",
        "Hardcoded sleep without rationale",
        "Print statement left in branch",
        "Long line over 100 columns",
        "Nested ternary harms readability",
        "Off-handed assertion message",
    ]
    findings = [_f("nit", t) for t in nit_titles]
    findings.append(_f("concern", "Critical race condition in handler"))
    result = _result(*findings)

    for strategy in ("exact", "title_substring"):
        out = apply_dedup(result, DistillConfig(dedup=strategy))  # type: ignore[arg-type]
        # All 21 survive — no false merge.
        assert len(out.findings) == 21
        # The concern is still in the list at its original severity.
        concerns = [f for f in out.findings if f.severity == "concern"]
        assert len(concerns) == 1
        assert concerns[0].title == "Critical race condition in handler"


def test_solo_finding_returns_identity():
    """A single finding can't duplicate anything; identity preserved
    so the transform composes cleanly with whatever runs next.
    """
    result = _result(_f("nit", "alone"))
    out = apply_dedup(result, DistillConfig(dedup="exact"))
    assert out is result


def test_severity_does_not_downgrade():
    """If the earlier finding is already at the higher severity, the
    later duplicate's lower severity does NOT downgrade it.
    """
    result = _result(_f("concern", "X"), _f("nit", "X"))
    out = apply_dedup(result, DistillConfig(dedup="exact"))
    assert len(out.findings) == 1
    assert out.findings[0].severity == "concern"


def test_unknown_severity_preserved_without_crash():
    """Severity is a trust-boundary label; a provider that emits an
    unknown value (typo, future-version label, corrupt JSON) must
    not crash the dedup transform when the finding happens to also
    duplicate another. Matches the existing severity-floor convention
    in ``reviewer/agent.py`` (see
    ``test_severity_floor_unknown_severity_in_finding_is_preserved``).

    Three duplicates: one normal nit, one with bogus severity, one
    concern. The concern's severity must still propagate to the
    survivor; the bogus row must not raise.
    """
    result = _result(
        _f("nit", "X"),
        _f("bogus_severity_value", "X"),  # type: ignore[arg-type]
        _f("concern", "X"),
    )
    out = apply_dedup(result, DistillConfig(dedup="exact"))
    assert len(out.findings) == 1
    assert out.findings[0].severity == "concern"


def test_unknown_survivor_severity_promotes_to_known_candidate():
    """Asymmetric edge case: when the *survivor's* existing severity
    is malformed but a later duplicate carries a parseable label,
    the merged group should inherit the parseable severity. The
    survivor's malformed label is unambiguously worse data than a
    known severity from the same merged group.

    Without this, a finding emitted with a typo'd severity would
    keep that typo even when subsequent duplicates carry valid
    labels — defeating the "survivor severity is the highest known
    severity in the group" contract.
    """
    result = _result(
        _f("bogus_severity_value", "X"),  # type: ignore[arg-type]
        _f("concern", "X"),
    )
    out = apply_dedup(result, DistillConfig(dedup="exact"))
    assert len(out.findings) == 1
    # Survivor's title/body still come from the earlier finding,
    # but its severity is promoted to the candidate's known label.
    assert out.findings[0].title == "X"
    assert out.findings[0].severity == "concern"
