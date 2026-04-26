"""Tests for ``reviewer.distill.transforms.severity_filter``.

Pins the contract:
- Default ``severity_floor="nit"`` is inert (identity).
- ``severity_floor="comment"`` drops nits, keeps comments + concerns.
- ``severity_floor="concern"`` keeps only concerns.
- Unknown severity strings are kept (matches the agent's existing
  ``test_severity_floor_unknown_severity_in_finding_is_preserved``
  convention).
- Feature preservation: an outlier concern in a sea of nits survives
  every floor — though this is structurally trivial for severity
  filtering (concern always outranks nit), the test pins it as
  documentation that the design principle holds.
- No-op cases return identity (``out is result``).
"""

from __future__ import annotations

from khonliang_reviewer import ReviewFinding, ReviewResult

from reviewer.distill.transforms.severity_filter import apply_severity_filter
from reviewer.rules.distill import DistillConfig


def _result(*findings: ReviewFinding) -> ReviewResult:
    return ReviewResult(request_id="req-test", summary="ok", findings=list(findings))


def _f(severity: str, title: str = "t", body: str = "b") -> ReviewFinding:
    return ReviewFinding(severity=severity, title=title, body=body)  # type: ignore[arg-type]


def test_default_floor_is_identity():
    """``severity_floor="nit"`` keeps every known severity, so the
    default-config path returns the same ``ReviewResult`` object —
    the inert-config invariant the pipeline shell relies on.
    """
    result = _result(_f("nit"), _f("comment"), _f("concern"))
    out = apply_severity_filter(result, DistillConfig())  # default floor "nit"
    assert out is result


def test_floor_comment_drops_nits():
    result = _result(_f("nit", "n1"), _f("comment", "c1"), _f("concern", "k1"))
    out = apply_severity_filter(result, DistillConfig(severity_floor="comment"))
    assert [f.title for f in out.findings] == ["c1", "k1"]


def test_floor_concern_keeps_only_concerns():
    result = _result(_f("nit"), _f("comment"), _f("concern", "the bug"))
    out = apply_severity_filter(result, DistillConfig(severity_floor="concern"))
    assert len(out.findings) == 1
    assert out.findings[0].title == "the bug"


def test_unknown_severity_is_preserved():
    """Severity is a trust-boundary label; an unknown value (typo,
    future-version label, corrupt JSON) is kept rather than dropped.
    Matches the agent's post-call filter convention
    (``test_severity_floor_unknown_severity_in_finding_is_preserved``).
    """
    result = _result(
        _f("bogus_severity_value", "keep me"),  # type: ignore[arg-type]
        _f("nit", "drop me"),
    )
    out = apply_severity_filter(result, DistillConfig(severity_floor="comment"))
    assert [f.title for f in out.findings] == ["keep me"]


def test_no_findings_below_floor_returns_identity():
    """When every finding already clears the floor, no actual filtering
    happened — return the same object so identity-equality holds and
    the transform composes cleanly with whatever runs next.
    """
    result = _result(_f("comment"), _f("concern"))
    out = apply_severity_filter(result, DistillConfig(severity_floor="comment"))
    assert out is result


def test_outlier_concern_survives_under_every_floor():
    """MS-B feature-preservation invariant: an outlier concern
    surrounded by 20 nits never drops, regardless of floor. The
    severity-filter case is structurally trivial (concern always
    outranks nit) but the test pins the invariant as documentation
    that the design principle holds across transforms.
    """
    findings = [_f("nit", f"nit-{i}") for i in range(20)]
    findings.append(_f("concern", "outlier concern"))
    result = _result(*findings)

    for floor in ("nit", "comment", "concern"):
        out = apply_severity_filter(
            result, DistillConfig(severity_floor=floor)  # type: ignore[arg-type]
        )
        concerns = [f for f in out.findings if f.severity == "concern"]
        assert len(concerns) == 1
        assert concerns[0].title == "outlier concern"


def test_empty_findings_returns_identity():
    """Edge case: a result with no findings is also identity-preserving
    so providers that emit empty review lists don't pay for a clone.
    """
    result = _result()
    out = apply_severity_filter(result, DistillConfig(severity_floor="concern"))
    assert out is result
