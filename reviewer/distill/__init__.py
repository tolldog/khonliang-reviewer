"""Post-provider distill pipeline.

Receives a :class:`khonliang_reviewer.ReviewResult` plus a
:class:`reviewer.rules.distill.DistillConfig` and returns a possibly-
shaped result. Transforms compose in a fixed order (dedup →
severity_filter → body_mode → max_findings). The ``dedup`` transform
ships in this module today (see
:mod:`reviewer.distill.transforms.dedup`); ``severity_filter``,
``body_mode``, and ``max_findings`` land in follow-up FRs. The
pipeline shell carries the ``audit_corpus`` audience short-circuit,
the call shape every transform plugs into, and an identity-preserving
guarantee on the inert-config path.

Consensus (``DistillConfig.consensus``) is *not* a transform — it
runs at the selector layer ahead of the provider call. The
``consensus`` field on ``DistillConfig`` is the rule-table → selector
signal; the distill pipeline never inspects it. See
``specs/MS-B/spec.md`` Open Question #2 resolution.

Likewise, ``dropped_findings`` (the audit-trail of findings the
pipeline removed) is a planned addition to ``ReviewResult`` in
``khonliang-reviewer-lib`` and lands in a follow-up FR. The current
transforms drop without recording, which preserves Acceptance #1's
behavior for ``audit_corpus`` (full short-circuit) but does not yet
satisfy Acceptance #5's ``dropped_findings == []`` invariant on
non-audit paths — that lands once the lib field is available.
"""

from __future__ import annotations

from khonliang_reviewer import ReviewResult

from reviewer.distill.transforms.dedup import apply_dedup
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

    Each transform is identity-preserving when its config slot is
    inert (e.g. ``dedup="none"``, future ``severity_floor="nit"``),
    so the default-config path through the pipeline returns the same
    ``ReviewResult`` object the provider produced.
    """
    if config.audience == "audit_corpus":
        return result
    result = apply_dedup(result, config)
    # Remaining transforms (severity_filter, body_mode, max_findings)
    # slot in here in subsequent FRs.
    return result


__all__ = ["run_pipeline"]
