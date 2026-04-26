"""Post-provider distill pipeline.

Receives a :class:`khonliang_reviewer.ReviewResult` plus a
:class:`reviewer.rules.distill.DistillConfig` and returns a possibly-
shaped result. Transforms compose in a fixed order (dedup →
severity_filter → body_mode → max_findings); each lands in its own
follow-up FR. This shell currently runs no transforms — it ships the
``audit_corpus`` audience short-circuit and the call shape that the
transforms-PRs will plug into.

Why the shell ships first: every transform-PR otherwise has to invent
its own integration point, which is the failure mode that motivated
the entire MS-B milestone (every quality FR re-litigating "where in
the call path does this run?"). Landing the rails before any cargo
keeps the transforms small + independent.

Consensus (``DistillConfig.consensus``) is *not* a transform — it
runs at the selector layer ahead of the provider call. The
``consensus`` field on ``DistillConfig`` is the rule-table → selector
signal; the distill pipeline never inspects it. See
``specs/MS-B/spec.md`` Open Question #2 resolution.

Likewise, ``dropped_findings`` (the audit-trail of findings the
pipeline removed) is a planned addition to ``ReviewResult`` in
``khonliang-reviewer-lib`` and lands in a follow-up FR. The shell
here returns the result unchanged so it does not depend on a
not-yet-shipped library field.
"""

from __future__ import annotations

from khonliang_reviewer import ReviewResult

from reviewer.rules.distill import DistillConfig


def run_pipeline(result: ReviewResult, config: DistillConfig) -> ReviewResult:
    """Apply the distill pipeline to ``result`` per ``config``.

    Pipeline order (from MS-B's spec; transforms land in subsequent
    FRs and slot in here in the order listed):

    1. ``dedup`` — collapse exact-text or title-substring duplicates.
    2. ``severity_filter`` — drop findings below ``config.severity_floor``.
    3. ``body_mode`` — shape ``summary`` + finding ``body`` length.
    4. ``max_findings`` — cap finding count.

    The ``audience == "audit_corpus"`` short-circuit returns the
    result unchanged regardless of any other config field. Audit /
    benchmark callers always see the raw provider output; that's
    the contract that lets aggressive shaping in other audiences
    stay safe (the raw output is recoverable by re-running with
    ``audit_corpus``).

    Until the transforms land, the non-short-circuit path is also
    a no-op — the shell ships first so subsequent FRs have a
    stable integration point.
    """
    if config.audience == "audit_corpus":
        return result
    # Transforms slot in here in subsequent FRs.
    return result


__all__ = ["run_pipeline"]
