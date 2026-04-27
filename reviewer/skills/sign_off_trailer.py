"""Format an ``Agent-Reviewed-by`` trailer line from a ``ReviewResult``.

Closes the trailer-format drift surface (`fr_reviewer_b846a19c`):
subagents have been hand-assembling the trailer line every commit,
and the format drifts (punctuation, ordering, whether the verdict
includes a reason). This module is the single source of truth for
the format; the agent's ``sign_off_trailer`` skill is a thin wrapper
that calls into :func:`build_trailer`.

The trailer shape (locked):

    Agent-Reviewed-by: <role>/<backend>/<model> <verdict>[: <reason>]

Where:

- ``role`` defaults to ``khonliang-reviewer`` (configurable for
  future cross-vendor reviewer roles).
- ``backend`` and ``model`` come from the ``ReviewResult`` (they
  carry the canonical names the provider used).
- ``verdict`` is computed from finding severity counts and the
  result's error_category — see :func:`compute_verdict`.
- ``reason`` is required for ``approved-with-findings`` and
  ``concerns-raised``; omitted for ``approved`` and
  ``escalated-approved``. Capped at ~80 chars so ``git log
  --oneline`` stays readable.

Parses cleanly under ``git interpret-trailers`` (the trailer key
``Agent-Reviewed-by`` is a custom key, intentionally distinct from
the standard ``Reviewed-by:`` so a future human reviewer's sign-off
doesn't collide).
"""

from __future__ import annotations

from typing import Any, Literal

from khonliang_reviewer import ReviewResult, severity_rank


Verdict = Literal[
    "approved",
    "approved-with-findings",
    "concerns-raised",
    "escalated-approved",
]


_REASON_MAX_CHARS = 80


def compute_verdict(result: ReviewResult) -> Verdict:
    """Map a ``ReviewResult`` to a four-value verdict.

    Mapping (per ``specs/MS-D/spec.md``):

    - ``error_category == "claude_cli_escalation"`` → ``escalated-approved``
      (the result came from the cross-vendor escalation path, so the
      sign-off records that vendor diversity).
    - Any concern-severity finding → ``concerns-raised``. The trailer
      is honest about the count; whether the caller chooses to merge
      anyway is a separate decision.
    - At least one comment- or nit-severity finding (with no concerns)
      → ``approved-with-findings``.
    - Zero findings → ``approved``.

    Findings with non-string or unrecognized severity are treated as
    concerns (kept loud) so a malformed-severity row never silently
    downgrades the verdict.
    """
    if result.error_category == "claude_cli_escalation":
        return "escalated-approved"

    counts = _severity_counts(result)
    if counts["concern"] > 0 or counts["unknown"] > 0:
        return "concerns-raised"
    if counts["comment"] > 0 or counts["nit"] > 0:
        return "approved-with-findings"
    return "approved"


def build_trailer(
    result: ReviewResult,
    *,
    role: str = "khonliang-reviewer",
    reason: str = "",
) -> dict[str, Any]:
    """Build a structured trailer record from a ``ReviewResult``.

    Returns ``{verdict, trailer_line}``. The caller stitches the
    trailer line into a commit message; the verdict is exposed as a
    structured field so subagents can branch on it.

    ``reason`` is auto-derived from the finding histogram when not
    supplied. An empty auto-reason for the ``approved`` and
    ``escalated-approved`` verdicts is intentional (no reason fits
    the trailer format for those cases). Caller-supplied ``reason``
    overrides the auto-derived one for both shapes.
    """
    verdict = compute_verdict(result)
    backend = result.backend or "unknown-backend"
    model = result.model or "unknown-model"

    if not reason:
        reason = _auto_reason(result, verdict)

    if reason and verdict in ("approved-with-findings", "concerns-raised"):
        reason = _truncate_reason(reason)
        trailer_line = (
            f"Agent-Reviewed-by: {role}/{backend}/{model} {verdict}: {reason}"
        )
    else:
        # approved / escalated-approved: no reason segment.
        # Or approved-with-findings / concerns-raised with no reason
        # auto-derived (rare; means findings list is empty but
        # severity counts said otherwise — should not happen).
        trailer_line = f"Agent-Reviewed-by: {role}/{backend}/{model} {verdict}"

    return {"verdict": verdict, "trailer_line": trailer_line}


def _severity_counts(result: ReviewResult) -> dict[str, int]:
    """Count findings by severity. ``unknown`` covers non-string and
    unparseable severity strings.
    """
    counts = {"nit": 0, "comment": 0, "concern": 0, "unknown": 0}
    for f in result.findings:
        severity = f.severity
        if not isinstance(severity, str):
            counts["unknown"] += 1
            continue
        try:
            severity_rank(severity)
        except ValueError:
            counts["unknown"] += 1
            continue
        # Severity is a known str — bucket it.
        counts[severity] = counts.get(severity, 0) + 1
    return counts


def _auto_reason(result: ReviewResult, verdict: Verdict) -> str:
    """Build a default reason from the finding histogram + first-concern.

    For ``concerns-raised`` the reason names the count and the first
    concern's title (or category) so ``git log --oneline`` shows the
    headline. For ``approved-with-findings`` the reason summarizes
    the count + severity mix. For ``approved`` / ``escalated-approved``
    the reason is empty (the trailer format omits the segment).
    """
    if verdict in ("approved", "escalated-approved"):
        return ""

    counts = _severity_counts(result)
    if verdict == "concerns-raised":
        concern_count = counts["concern"] + counts["unknown"]
        first_concern = next(
            (f for f in result.findings if f.severity == "concern"), None
        )
        # Fall back to first-finding when no concern-severity finding
        # is present (shouldn't happen for this verdict; defensive).
        if first_concern is None:
            first_concern = result.findings[0] if result.findings else None
        if first_concern is None:
            return f"{concern_count} concern" + ("s" if concern_count != 1 else "")
        # Prefer a parseable category (e.g. "race_condition") when
        # the provider supplied one — short and stable for log
        # scanning. Falls back to the title otherwise.
        anchor = first_concern.category or first_concern.title or "(no title)"
        plural = "s" if concern_count != 1 else ""
        return f"{concern_count} concern{plural}: {anchor}"

    # approved-with-findings: enumerate non-zero severity buckets.
    parts = []
    if counts["comment"]:
        parts.append(f"{counts['comment']} comment" + ("s" if counts["comment"] != 1 else ""))
    if counts["nit"]:
        parts.append(f"{counts['nit']} nit" + ("s" if counts["nit"] != 1 else ""))
    return " + ".join(parts) if parts else ""


def _truncate_reason(reason: str) -> str:
    """Cap the reason segment so the trailer stays readable in
    ``git log --oneline`` (which already truncates at terminal width).
    """
    if len(reason) <= _REASON_MAX_CHARS:
        return reason
    return reason[: _REASON_MAX_CHARS - 1].rstrip() + "…"


__all__ = ["Verdict", "build_trailer", "compute_verdict"]
