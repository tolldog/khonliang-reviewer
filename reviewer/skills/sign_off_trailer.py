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
  ``escalated-approved`` (those verdicts produce a clean two-
  segment trailer). Caller-supplied reasons are honored on every
  verdict — the auto-reason logic only fires when the caller
  doesn't supply one. Reason is capped at ~80 chars so ``git log
  --oneline`` stays readable.

- ``approved-with-findings`` reasons follow the spec's locked
  ``<histogram> filtered`` shape — e.g. ``2 nits filtered``,
  ``1 comment + 2 nits filtered``. The ``filtered`` suffix is a
  participial adjective on the histogram meaning "flagged but
  non-blocking"; the numeric counts are the surviving findings.
  ``UsageEvent.findings_filtered_count`` (the count
  severity_filter dropped) is telemetry, not trailer copy — the
  trailer talks about what survived, not what was dropped.

Errored review results (``disposition == "errored"``) raise
:class:`ValueError` from :func:`compute_verdict` /
:func:`build_trailer` rather than producing a trailer — committing
an ``Agent-Reviewed-by`` line for a review that didn't actually
run is misleading sign-off provenance. Agent handlers catch the
exception and surface ``{error: ...}`` to the caller. The
``claude_cli_escalation`` case is NOT an errored disposition (it's
a successful cross-vendor escalation result that happens to set
``error_category`` for routing purposes); it still maps to
``escalated-approved`` per spec.

All trailer segments (role, backend, model, reason) are sanitized
against newline characters before interpolation so a malicious or
malformed provider output can't inject additional trailer lines
into the commit message.

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

    Raises :class:`ValueError` when the result's disposition is
    ``"errored"`` — committing a sign-off trailer for a review
    that didn't actually run would be misleading provenance. The
    ``claude_cli_escalation`` case is checked first so a
    successful escalation result (which sets ``error_category``
    for routing but is NOT disposition=errored) still maps to
    ``escalated-approved``.
    """
    if result.error_category == "claude_cli_escalation":
        return "escalated-approved"
    if result.disposition == "errored":
        message = result.error or "(no error message)"
        raise ValueError(
            f"cannot format sign-off trailer for errored review: {message}"
        )

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
    # role/backend/model are joined by "/" in the trailer's locked
    # ``role/backend/model`` shape, so a "/" inside any of them would
    # add phantom path segments and break unambiguous parsing.
    # Model ids in this repo can carry "/" (see
    # tests/test_benchmark_sweep.py's "kimi-k2/5/cloud" example);
    # _sanitize_path_segment normalizes those to "-" alongside the
    # newline-strip every segment gets.
    backend = _sanitize_path_segment(result.backend or "unknown-backend")
    model = _sanitize_path_segment(result.model or "unknown-model")
    role = _sanitize_path_segment(role)

    if not reason:
        reason = _auto_reason(result, verdict)
    # ``reason`` keeps "/" — operator-supplied reasons often carry
    # path references like "reviewer/agent.py:42" that should
    # survive intact. Newlines / tabs are still collapsed (the
    # injection-prevention concern).
    reason = _sanitize_segment(reason)

    # The trailer emits the ``: <reason>`` segment whenever a
    # non-empty reason exists, regardless of verdict. The spec
    # describes the reason as REQUIRED for approved-with-findings /
    # concerns-raised but only OPTIONAL for approved /
    # escalated-approved — the auto-reason logic returns empty for
    # the latter cases unless ``findings_filtered_count`` is > 0,
    # in which case the trailer still surfaces the filtered-count
    # so subagents see that severity_filter shaped the payload.
    if reason:
        reason = _truncate_reason(reason)
        trailer_line = (
            f"Agent-Reviewed-by: {role}/{backend}/{model} {verdict}: {reason}"
        )
    else:
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
    *contributing* finding's title (or category) so ``git log
    --oneline`` shows the headline. "Contributing" = severity is
    ``concern`` OR an unparseable / non-string severity (since
    unknown rows count toward concerns per ``compute_verdict``);
    leading nit/comment rows are skipped so the anchor doesn't
    accidentally point at a low-severity finding.

    For ``approved-with-findings`` the reason follows the spec's
    locked "<histogram> filtered" shape — e.g. ``2 nits filtered``,
    ``1 comment + 2 nits filtered``. The ``filtered`` suffix is a
    participial adjective on the histogram meaning "flagged but
    non-blocking". The numeric counts are the surviving findings
    in ``result.findings``; the spec deliberately doesn't reference
    ``UsageEvent.findings_filtered_count`` here (that field is
    telemetry, not trailer copy).

    For ``approved`` / ``escalated-approved`` the reason is empty —
    the trailer omits the segment so a clean approval reads
    cleanly.
    """
    counts = _severity_counts(result)

    if verdict == "concerns-raised":
        concern_count = counts["concern"] + counts["unknown"]
        anchor_finding = _first_concern_or_unknown(result)
        if anchor_finding is None:
            return (
                f"{concern_count} concern"
                + ("s" if concern_count != 1 else "")
            )
        anchor = (
            anchor_finding.category
            or anchor_finding.title
            or "(no title)"
        )
        plural = "s" if concern_count != 1 else ""
        return f"{concern_count} concern{plural}: {anchor}"

    if verdict == "approved-with-findings":
        parts = []
        if counts["comment"]:
            parts.append(
                f"{counts['comment']} comment"
                + ("s" if counts["comment"] != 1 else "")
            )
        if counts["nit"]:
            parts.append(
                f"{counts['nit']} nit" + ("s" if counts["nit"] != 1 else "")
            )
        if not parts:
            return ""
        return " + ".join(parts) + " filtered"

    # approved / escalated-approved.
    return ""


def _first_concern_or_unknown(result: ReviewResult):
    """Return the first finding whose severity counts toward
    ``concerns-raised`` — i.e. severity ``concern`` OR an
    unparseable / non-string severity. Skips leading nit/comment
    rows so the anchor reflects what actually triggered the verdict.
    Returns ``None`` when no contributing row exists (defensive;
    shouldn't fire for this verdict).
    """
    for f in result.findings:
        severity = f.severity
        if severity == "concern":
            return f
        if not isinstance(severity, str):
            return f
        try:
            severity_rank(severity)
        except ValueError:
            return f
    return None


def _sanitize_segment(value: str) -> str:
    """Collapse whitespace + strip newlines from a trailer segment.

    The trailer format is single-line. A malicious or malformed
    provider output that injects ``\\n`` / ``\\r`` into a finding
    title (which feeds the auto-reason anchor) or a backend / model
    string could otherwise forge additional trailer keys in the
    commit message — e.g. an attacker-supplied finding titled
    ``hack\\nApproved-by: someone-else`` would produce a trailer
    block claiming a second sign-off. Replace any whitespace
    sequence (including CR/LF and tabs) with a single space and
    strip the result.

    Idempotent: re-sanitizing already-clean input is a no-op.
    """
    if not value:
        return value
    # Collapse every whitespace run (CR, LF, tab, multiple spaces)
    # into a single space, then strip leading/trailing space.
    return " ".join(value.split())


def _sanitize_path_segment(value: str) -> str:
    """Sanitize a segment that's joined into the locked
    ``role/backend/model`` path of the trailer line.

    Stricter than :func:`_sanitize_segment`: in addition to the
    newline / whitespace normalization, replaces ``/``, ``\\``,
    and the remaining single-space characters with ``-`` so the
    triple stays a single space-delimited token in the locked
    ``Agent-Reviewed-by: <role>/<backend>/<model> <verdict>...``
    format. Operator-supplied roles, provider-reported backend
    names, and model ids all run through this — model ids carry
    the most variability (e.g. ``kimi-k2/5/cloud``).
    """
    cleaned = _sanitize_segment(value)
    if not cleaned:
        return cleaned
    # Translate path-separator chars AND any remaining spaces to
    # ``-``. Preserves digits, dots, colons, underscores so
    # canonical model ids stay readable in the trailer.
    return (
        cleaned
        .replace("/", "-")
        .replace("\\", "-")
        .replace(" ", "-")
    )


def _truncate_reason(reason: str) -> str:
    """Cap the reason segment so the trailer stays readable in
    ``git log --oneline`` (which already truncates at terminal width).
    """
    if len(reason) <= _REASON_MAX_CHARS:
        return reason
    return reason[: _REASON_MAX_CHARS - 1].rstrip() + "…"


__all__ = ["Verdict", "build_trailer", "compute_verdict"]
