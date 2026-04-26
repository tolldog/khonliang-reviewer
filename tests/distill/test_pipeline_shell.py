"""Tests for the post-provider distill-pipeline shell.

Pin the contract every transform-PR depends on:

- ``audit_corpus`` audience returns the result unchanged regardless
  of any other config field (the short-circuit fires before any
  transform runs).
- The inert-config default path is identity-preserving — every
  transform must return the same object on its inert slot
  (``dedup="none"``, ``severity_floor="nit"``, etc.) so a
  default-config review never pays for a no-op clone.
- The function signature is ``(ReviewResult, DistillConfig) -> ReviewResult``.

Active-transform behavior is tested in each transform's own
``test_<transform>.py`` (e.g. ``test_dedup.py``).
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

    # Full identity: audit_corpus is the contract that lets aggressive
    # shaping in other audiences stay safe (raw output is recoverable).
    # Asserting object identity catches accidental clones / shaping of
    # any ReviewResult field (usage / disposition / backend / model /
    # summary / findings) — anything weaker would let drift slip
    # through silently as transforms land.
    assert out is result


def test_default_config_path_is_identity_preserving():
    """Inert-config invariant: every transform must return the same
    object on its inert slot (``dedup="none"``, ``severity_floor="nit"``,
    ``body_mode="full"``, ``max_findings=None``) so a default-config
    review never pays for a no-op clone. The default ``DistillConfig()``
    sets every slot inert by design — a misconfigured rule never
    silently shapes findings.

    Future transform PRs MUST keep returning ``result is`` on their
    inert slot or this assertion trips.
    """
    result = _result_with_findings("nit", "concern")
    config = DistillConfig()  # all slots inert by default

    out = run_pipeline(result, config)

    assert out is result


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
