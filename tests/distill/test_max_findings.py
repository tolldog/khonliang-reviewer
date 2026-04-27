"""Tests for ``reviewer.distill.transforms.max_findings``.

Pins the contract:
- ``max_findings=None`` is identity (default; misconfigured rule
  never silently caps).
- When ``cap >= len(findings)`` and the input is already in
  severity-desc order: identity.
- When the cap actually drops findings: highest-severity items
  survive.
- Stable sort: among same-severity ties, first-occurrence order
  wins.
- Unknown severity strings are treated as max-rank so a malformed
  finding isn't silently dropped by the cap.
- Outlier preservation: an outlier concern surrounded by 20
  same-severity nits with cap=5 still survives — the spec's
  10×-outlier feature-preservation invariant.
"""

from __future__ import annotations

import pytest
from khonliang_reviewer import ReviewFinding, ReviewResult

from reviewer.distill.transforms.max_findings import apply_max_findings
from reviewer.rules.distill import DistillConfig


def _result(*findings: ReviewFinding) -> ReviewResult:
    return ReviewResult(request_id="req-test", summary="ok", findings=list(findings))


def _f(severity: str, title: str) -> ReviewFinding:
    return ReviewFinding(severity=severity, title=title, body="b")  # type: ignore[arg-type]


def test_none_cap_is_identity():
    """Default ``max_findings=None`` keeps every finding regardless of
    order — the inert-config invariant the pipeline shell depends on.
    """
    result = _result(_f("nit", "n"), _f("concern", "c"))
    out = apply_max_findings(result, DistillConfig())  # max_findings=None default
    assert out is result


def test_already_sorted_under_cap_is_identity():
    """When the findings are already in severity-desc order and the
    cap is >= count, neither sort nor truncate changes anything;
    identity is preserved.
    """
    result = _result(_f("concern", "c"), _f("comment", "m"), _f("nit", "n"))
    out = apply_max_findings(result, DistillConfig(max_findings=10))
    assert out is result


def test_cap_drops_lowest_severity():
    """``cap < len`` drops the lowest-severity findings first.
    Stable sort preserves first-occurrence order for ties.
    """
    result = _result(
        _f("nit", "n1"),
        _f("nit", "n2"),
        _f("concern", "c1"),
        _f("comment", "m1"),
    )
    out = apply_max_findings(result, DistillConfig(max_findings=2))
    # Severity desc: concern, comment, nit, nit. Cap=2 keeps
    # concern + comment.
    assert [f.title for f in out.findings] == ["c1", "m1"]


def test_stable_sort_preserves_first_occurrence_among_ties():
    """When the cap leaves room for ties, the earliest finding
    among each rank survives.
    """
    result = _result(
        _f("nit", "n-A"),
        _f("comment", "m-A"),
        _f("nit", "n-B"),
        _f("comment", "m-B"),
    )
    # Severity desc, stable: m-A, m-B, n-A, n-B. Cap=2 keeps m-A + m-B.
    out = apply_max_findings(result, DistillConfig(max_findings=2))
    assert [f.title for f in out.findings] == ["m-A", "m-B"]


def test_sort_only_when_cap_set():
    """A ``max_findings`` value reorders findings even when no cap
    truncation occurs, because the spec says "sort then truncate"
    is one operation. Provider-emission order is the tie-breaker
    among equal severities.
    """
    result = _result(_f("nit", "n"), _f("concern", "c"), _f("comment", "m"))
    # Cap >= len, but the findings are NOT in severity-desc order,
    # so the transform does reorder them.
    out = apply_max_findings(result, DistillConfig(max_findings=10))
    assert [f.title for f in out.findings] == ["c", "m", "n"]


def test_outlier_concern_survives_under_tight_cap():
    """MS-B feature-preservation invariant: an outlier concern
    surrounded by 20 nits and a cap of 5 still survives. The
    severity-desc sort guarantees the concern is first; the cap
    keeps it plus the next 4 (provider-order-preserved) nits.
    """
    findings = [_f("nit", f"nit-{i}") for i in range(20)]
    findings.insert(10, _f("concern", "outlier concern"))  # mid-list
    result = _result(*findings)
    out = apply_max_findings(result, DistillConfig(max_findings=5))
    titles = [f.title for f in out.findings]
    assert "outlier concern" in titles
    assert titles[0] == "outlier concern"  # severity desc puts it first
    assert len(out.findings) == 5


def test_unknown_severity_treated_as_max_rank():
    """A finding with an unknown severity string isn't silently
    dropped by the cap — it sorts at max-rank (concern-equivalent)
    so it survives a tight cap. Matches the keep-on-unknown
    convention used by severity_filter and dedup's _bumped helper.
    """
    result = _result(
        _f("nit", "n1"),
        _f("bogus_severity_value", "kept-bogus"),  # type: ignore[arg-type]
        _f("concern", "kept-concern"),
    )
    out = apply_max_findings(result, DistillConfig(max_findings=2))
    titles = [f.title for f in out.findings]
    # Both unknown and concern sort to max-rank; stable sort keeps
    # original order between them. Cap=2 keeps both, drops the nit.
    assert "kept-bogus" in titles
    assert "kept-concern" in titles
    assert "n1" not in titles


def test_empty_findings_returns_identity():
    """Edge case: a result with no findings is identity-preserving
    even with a cap set, so providers emitting empty review lists
    don't pay for a clone.
    """
    result = _result()
    out = apply_max_findings(result, DistillConfig(max_findings=5))
    assert out is result


def test_cap_exactly_zero_keeps_no_findings():
    """``max_findings=0`` is an aggressive cap — zero findings
    survive. Edge case worth pinning so future operators can
    rely on the cap meaning literally what it says.
    """
    result = _result(_f("concern", "c"))
    out = apply_max_findings(result, DistillConfig(max_findings=0))
    assert out.findings == []


def test_negative_cap_raises():
    """Negative caps would silently slice from the END rather than
    truncate (Python slicing semantics: ``findings[:-2]`` drops the
    last two rather than capping to zero). Raise loudly so a
    misconfigured rule never silently reshapes results — matches
    the ValueError contract used by dedup's 'semantic' strategy
    and body_mode's unknown-mode case.
    """
    result = _result(_f("nit", "a"), _f("nit", "b"), _f("nit", "c"))
    with pytest.raises(ValueError, match="max_findings"):
        apply_max_findings(result, DistillConfig(max_findings=-1))
