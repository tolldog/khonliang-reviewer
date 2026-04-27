"""Max-findings transform — sort by severity desc + first-position, then cap.

Last post-provider transform in the distill pipeline. From the
MS-B spec: "sort findings by severity desc + first-position then
truncate". When the cap kicks in we want the highest-severity
findings to survive, not the first N the provider happened to emit.

Behavior per ``DistillConfig.max_findings``:

- ``None`` (default): identity. No sort, no truncate. The
  default-config inert-pipeline invariant relies on this.
- ``cap >= len(findings)``: stable-sort by severity desc; identity
  when the sort doesn't actually reorder. No truncation needed.
- ``cap < len(findings)``: stable-sort by severity desc, truncate
  to ``cap``.

Unknown severity strings are treated as max-rank ("concern"-equivalent)
so a finding with a malformed severity label survives the cap rather
than being silently dropped. Matches the keep-on-unknown convention
used by ``apply_severity_filter`` and the dedup transform's
``_bumped`` helper. Stable sort preserves first-occurrence order
among ties.

Identity preservation: returns the same ``ReviewResult`` object when
the sorted+truncated findings list is identical (in order and content)
to the input. The pipeline shell's inert-config invariant relies on
this for the default ``max_findings=None`` and for any payload that's
already sorted under any cap.
"""

from __future__ import annotations

from dataclasses import replace

from khonliang_reviewer import ReviewFinding, ReviewResult, severity_rank

from reviewer.rules.distill import DistillConfig


# Resolved once at import — the rank a finding with an unknown
# severity is assigned so it survives the cap (treated as the most
# severe known label). See module docstring.
_UNKNOWN_SEVERITY_RANK = severity_rank("concern")


def apply_max_findings(result: ReviewResult, config: DistillConfig) -> ReviewResult:
    """Sort by severity desc and cap to ``config.max_findings``.

    Returns the same ``ReviewResult`` object when no findings would
    actually move or drop (so identity-equality holds in the inert
    cases — ``max_findings=None`` always, plus any payload that's
    already sorted within a cap that's >= its length).
    """
    cap = config.max_findings
    if cap is None:
        return result
    if cap < 0:
        # A negative cap would silently flip Python's slicing
        # semantics (``findings[:-2]`` drops from the END rather
        # than capping to zero), which would surprise every caller
        # that ever supplies a misconfigured rule. Fail loudly so
        # the misconfiguration shows up immediately — matches the
        # ValueError contract used by dedup's 'semantic' strategy
        # and body_mode's unknown-mode case.
        raise ValueError(
            f"DistillConfig.max_findings={cap!r} must be >= 0 or None; "
            "negative caps would slice from the end rather than truncate."
        )

    # Negate the rank so higher-severity sorts first; stable sort
    # preserves first-occurrence order among equal-rank findings.
    sorted_findings = sorted(result.findings, key=_sort_key)
    truncated = sorted_findings[:cap] if cap < len(sorted_findings) else sorted_findings

    # Identity-preserve when the (sorted, truncated) list is the same
    # tuple of objects in the same order as the input. ``is`` checks
    # element-wise so a sort that didn't reorder anything still
    # returns identity.
    if len(truncated) == len(result.findings) and all(
        a is b for a, b in zip(truncated, result.findings)
    ):
        return result
    return replace(result, findings=truncated)


def _sort_key(f: ReviewFinding) -> int:
    """Return ``-rank`` so higher severity sorts first under
    ``sorted``. Unknown severity strings AND non-string severities
    use the max-rank fallback so malformed provider output doesn't
    silently drop the finding or crash the pipeline. Matches
    ``apply_severity_filter``'s trust-boundary handling: severity
    is a Literal type-checker contract, but the bus boundary can
    deliver wider payloads (``None``, list, int, etc.) that the
    type system doesn't enforce.
    """
    severity = f.severity
    if not isinstance(severity, str):
        return -_UNKNOWN_SEVERITY_RANK
    try:
        return -severity_rank(severity)
    except ValueError:
        return -_UNKNOWN_SEVERITY_RANK


__all__ = ["apply_max_findings"]
