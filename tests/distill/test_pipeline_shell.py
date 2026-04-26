"""Tests for the post-provider distill-pipeline shell.

The shell currently runs no transforms — those land in subsequent
FRs. The tests pin the contract that lets transform-PRs land
without re-litigating the integration shape:

- ``audit_corpus`` audience returns the result unchanged.
- Default audience returns the result unchanged today (will gain
  transform behavior as each transform lands).
- The function signature is ``(ReviewResult, DistillConfig) -> ReviewResult``.
"""

from __future__ import annotations

from khonliang_reviewer import ReviewFinding, ReviewResult

from reviewer.distill import run_pipeline
from reviewer.rules.distill import DistillConfig


def _result_with_findings(*severities: str) -> ReviewResult:
    """Build a ReviewResult carrying findings of the given severities.

    Helper so each test reads as ``what shape went in / what shape
    came out`` rather than dragging boilerplate.
    """
    return ReviewResult(
        request_id="req-test",
        summary="ok",
        findings=[
            ReviewFinding(severity=s, title=f"f{i}", body="b")  # type: ignore[arg-type]
            for i, s in enumerate(severities)
        ],
    )


def test_audit_corpus_short_circuits_with_unchanged_findings():
    """``audit_corpus`` is the audit / benchmark path — it must
    return the raw provider output regardless of any other config
    field. This is the contract that lets aggressive shaping in
    other audiences stay safe (the raw output is recoverable by
    re-running with ``audit_corpus``).
    """
    result = _result_with_findings("nit", "comment", "concern")
    config = DistillConfig(
        audience="audit_corpus",
        # These fields would normally trim/filter findings, but
        # audit_corpus must override all of them:
        severity_floor="concern",
        max_findings=1,
        body_mode="compact",
        dedup="exact",
    )

    out = run_pipeline(result, config)

    assert [f.severity for f in out.findings] == ["nit", "comment", "concern"]
    assert out.summary == result.summary


def test_default_audience_returns_result_today():
    """Pre-transforms, the default-audience path is also a no-op.
    This test will start failing — by design — as each transform
    lands and starts shaping the result. At that point each
    transform PR is responsible for updating this test's expectations.
    Until then it pins "the shell is wired in but inert".
    """
    result = _result_with_findings("nit", "concern")
    config = DistillConfig()  # audience defaults to agent_consumption

    out = run_pipeline(result, config)

    assert out is result or out.findings == result.findings


def test_run_pipeline_signature_is_stable():
    """Pins ``(ReviewResult, DistillConfig) -> ReviewResult`` as the
    transforms-PR contract. Each follow-up transform plugs in via
    this signature; renaming the function or its kwargs would break
    every transform PR's wiring at once.
    """
    import inspect

    sig = inspect.signature(run_pipeline)
    params = list(sig.parameters.values())
    assert len(params) == 2
    assert params[0].name == "result"
    assert params[1].name == "config"
